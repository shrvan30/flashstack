"""OpenAI-compatible FastAPI server.

`GET /v1/models`, `POST /v1/chat/completions` (streaming and not), `GET /metrics`.
All GPU work is serialised through the scheduler's single loop, so the handlers
never touch the runner directly.

Metrics reporting has one unavoidable asymmetry. A non-streaming response carries
both `x-ttft-ms` and `x-decode-tps` headers, because the whole generation is done
before the response starts. A streaming response can carry `x-ttft-ms` — the
handler waits for the first token before returning, so the value is real — but
not `x-decode-tps`, since HTTP headers precede the body and decode throughput is
only known once the last token has been sent. Streaming therefore reports both
figures in the final chunk's `metrics` field instead.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from engine.sampling import SamplingParams
from server.scheduler import GenerationRequest, Scheduler
from server.schemas import (
    SSE_DONE,
    ChatCompletionRequest,
    ModelCard,
    ModelList,
    chunk,
    completion_response,
    make_usage,
    sse,
)

logger = logging.getLogger("flashstack.server")

MODEL_ENV = "FLASHSTACK_MODEL"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_runner(model_name: str):
    """Pick a runner from the model id. GPT-2 and Qwen2.5 are the supported families."""
    lowered = model_name.lower()
    if lowered.startswith("gpt2") or lowered.startswith("openai-community/gpt2"):
        from engine.models.gpt2 import GPT2Runner

        return GPT2Runner(model_name)
    from engine.models.qwen25 import Qwen25Runner

    return Qwen25Runner(model_name)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # uvicorn configures only its own loggers, so without this the scheduler's
    # batch lines are emitted and then dropped. Those lines are the evidence that
    # batching happens at all, so they have to reach the operator's console.
    if not logging.getLogger("flashstack").handlers:
        logging.basicConfig(
            level=os.environ.get("FLASHSTACK_LOG_LEVEL", "INFO"),
            format="%(levelname)s:     %(name)s - %(message)s",
        )
    logging.getLogger("flashstack").setLevel(
        os.environ.get("FLASHSTACK_LOG_LEVEL", "INFO")
    )

    model_name = os.environ.get(MODEL_ENV, DEFAULT_MODEL)
    logger.info("loading %s", model_name)
    runner = build_runner(model_name)
    scheduler = Scheduler(
        runner,
        max_batch=int(os.environ.get("FLASHSTACK_MAX_BATCH", "4")),
        window_ms=float(os.environ.get("FLASHSTACK_BATCH_WINDOW_MS", "25")),
    )
    await scheduler.start()

    app.state.model_name = model_name
    app.state.runner = runner
    app.state.scheduler = scheduler
    logger.info(
        "ready: %s, %d layers, %d query heads, %d kv heads, cache %.0f MiB",
        model_name,
        runner.num_layers,
        runner.num_heads,
        runner.num_kv_heads,
        runner.cache.memory_bytes() / 2**20,
    )
    try:
        yield
    finally:
        await scheduler.stop()


app = FastAPI(title="flashstack", version="0.2.0", lifespan=lifespan)


@app.get("/v1/models")
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id=app.state.model_name)])


@app.get("/metrics")
async def metrics() -> JSONResponse:
    return JSONResponse(app.state.scheduler.metrics.snapshot())


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": app.state.model_name}


def _to_sampling_params(request: ChatCompletionRequest, runner) -> SamplingParams:
    return SamplingParams(
        temperature=request.temperature if request.temperature is not None else 0.0,
        top_p=request.top_p if request.top_p is not None else 1.0,
        max_tokens=request.max_tokens if request.max_tokens is not None else 128,
        stop_token_ids=set(runner.eos_token_ids),
        seed=request.seed,
    )


def _encode_prompt(runner, request: ChatCompletionRequest) -> torch.Tensor:
    messages = [message.model_dump() for message in request.messages]
    if hasattr(runner, "apply_chat_template"):
        return runner.apply_chat_template(messages)
    # GPT-2 has no chat template; flatten to plain text so the endpoint still works.
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    return runner.tokenizer(text, return_tensors="pt").input_ids[0].to(runner.device)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    runner = app.state.runner
    scheduler: Scheduler = app.state.scheduler

    prompt_ids = _encode_prompt(runner, request)
    if prompt_ids.shape[0] >= runner.cache.max_seq:
        raise HTTPException(
            status_code=400,
            detail=(
                f"prompt of {prompt_ids.shape[0]} tokens leaves no room in a "
                f"{runner.cache.max_seq}-token cache"
            ),
        )

    generation = GenerationRequest(
        prompt_ids=prompt_ids,
        params=_to_sampling_params(request, runner),
        model=request.model,
    )
    await scheduler.submit(generation)

    if request.stream:
        return await _stream(generation, runner)
    return await _collect(generation, runner)


async def _collect(generation: GenerationRequest, runner) -> JSONResponse:
    while await generation.tokens.get() is not None:
        pass
    if generation.error is not None:
        raise HTTPException(status_code=503, detail=str(generation.error))

    text = runner.tokenizer.decode(generation.generated, skip_special_tokens=True)
    body = completion_response(
        request_id=generation.request_id,
        model=generation.model,
        content=text,
        finish_reason=generation.finish_reason or "stop",
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        created=generation.created,
    )
    return JSONResponse(
        body,
        headers={
            "x-ttft-ms": f"{generation.ttft_ms:.2f}",
            "x-decode-tps": f"{generation.decode_tps:.2f}",
        },
    )


async def _stream(generation: GenerationRequest, runner) -> StreamingResponse:
    # Wait for the first token here rather than inside the generator: it makes
    # x-ttft-ms a real measurement on a streaming response, since response
    # headers are sent when the generator's first chunk is yielded.
    first = await generation.tokens.get()
    if generation.error is not None:
        raise HTTPException(status_code=503, detail=str(generation.error))

    async def events() -> AsyncIterator[str]:
        yield sse(
            chunk(
                generation.request_id,
                generation.model,
                generation.created,
                delta={"role": "assistant", "content": ""},
            )
        )

        token = first
        emitted = 0
        while token is not None:
            text = runner.tokenizer.decode([token], skip_special_tokens=True)
            if text:
                yield sse(
                    chunk(
                        generation.request_id,
                        generation.model,
                        generation.created,
                        delta={"content": text},
                    )
                )
            emitted += 1
            token = await generation.tokens.get()

        yield sse(
            chunk(
                generation.request_id,
                generation.model,
                generation.created,
                delta={},
                finish_reason=generation.finish_reason or "stop",
                usage=make_usage(generation.prompt_tokens, generation.completion_tokens),
                metrics={
                    "ttft_ms": round(generation.ttft_ms, 2),
                    "decode_tps": round(generation.decode_tps, 2),
                },
            )
        )
        yield SSE_DONE

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "x-ttft-ms": f"{generation.ttft_ms:.2f}",
            "cache-control": "no-cache",
            "x-accel-buffering": "no",
        },
    )
