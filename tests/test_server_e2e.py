"""End-to-end server smoke test against a real model on the GPU.

Runs the app in-process with a live scheduler and drives it with the unmodified
`openai` client, which is the Phase 3 acceptance criterion: the point is not that
our own code can talk to our own server, but that a client written against
OpenAI's API can.
"""

from __future__ import annotations

import asyncio
import os

import pytest

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def live_server():
    """Start uvicorn on a free port in a background thread; yield its base URL."""
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    os.environ["FLASHSTACK_MODEL"] = MODEL

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    from server.app import app

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if httpx.get(f"{base}/health", timeout=1.0).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:  # pragma: no cover
        server.should_exit = True
        pytest.fail("server did not become healthy in time")

    yield base

    server.should_exit = True
    thread.join(timeout=30)


@pytest.fixture(scope="module")
def client(live_server):
    from openai import OpenAI

    return OpenAI(base_url=f"{live_server}/v1", api_key="not-needed")


@pytest.mark.gpu
def test_models_endpoint_lists_the_loaded_model(client):
    models = client.models.list()
    assert [m.id for m in models.data] == [MODEL]


@pytest.mark.gpu
def test_non_streamed_chat_completion(client):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        max_tokens=32,
    )
    content = response.choices[0].message.content
    print(f"\nnon-streamed: {content!r}")

    assert content, "the model returned no text"
    assert "Paris" in content, f"expected a correct answer, got {content!r}"
    assert response.choices[0].finish_reason in {"stop", "length"}
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert (
        response.usage.total_tokens
        == response.usage.prompt_tokens + response.usage.completion_tokens
    )


@pytest.mark.gpu
def test_streamed_chat_completion_yields_incremental_deltas(client):
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Name three colours."}],
        max_tokens=40,
        stream=True,
    )

    deltas, final = [], None
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            deltas.append(event.choices[0].delta.content)
        if event.choices and event.choices[0].finish_reason is not None:
            final = event

    text = "".join(deltas)
    print(f"\nstreamed ({len(deltas)} deltas): {text!r}")

    assert len(deltas) > 1, "a stream should arrive in more than one piece"
    assert text.strip()
    assert final is not None, "the stream never reported a finish_reason"


@pytest.mark.gpu
def test_streaming_and_non_streaming_agree_at_temperature_zero(client):
    messages = [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]

    whole = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=16
    ).choices[0].message.content

    stream = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=16, stream=True
    )
    streamed = "".join(
        e.choices[0].delta.content
        for e in stream
        if e.choices and e.choices[0].delta.content
    )

    print(f"\nnon-streamed {whole!r} vs streamed {streamed!r}")
    assert whole == streamed, "the two transports disagree on a deterministic request"


@pytest.mark.gpu
def test_concurrent_requests_share_a_decode_batch(live_server):
    """The static-batching claim, asserted rather than read off a log line."""
    import httpx

    async def one(index: int) -> int:
        async with httpx.AsyncClient(timeout=180) as http:
            response = await http.post(
                f"{live_server}/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": f"Count to {index + 3}."}],
                    "max_tokens": 24,
                },
            )
            return response.status_code

    async def drive() -> list[int]:
        return await asyncio.gather(*(one(i) for i in range(4)))

    import httpx as _httpx

    before = _httpx.get(f"{live_server}/metrics").json()["batching"]
    statuses = asyncio.run(drive())
    after = _httpx.get(f"{live_server}/metrics").json()["batching"]

    print(f"\nbatching before={before} after={after}")
    assert statuses == [200] * 4
    assert after["max_batch_size"] >= 2, (
        "four concurrent requests never shared a decode batch; "
        "static batching is not working"
    )
    assert (
        after["batches_with_multiple_requests"]
        > before["batches_with_multiple_requests"]
    )


@pytest.mark.gpu
def test_metrics_endpoint_reports_real_numbers(live_server):
    import httpx

    snapshot = httpx.get(f"{live_server}/metrics").json()
    print(f"\nmetrics: {snapshot}")
    assert snapshot["total_requests"] > 0
    assert snapshot["ttft_ms"]["p50"] > 0
    assert snapshot["decode_tps"]["mean"] > 0


@pytest.mark.gpu
def test_response_carries_timing_headers(live_server):
    import httpx

    response = httpx.post(
        f"{live_server}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
        },
        timeout=180,
    )
    assert response.status_code == 200
    assert float(response.headers["x-ttft-ms"]) > 0
    assert float(response.headers["x-decode-tps"]) > 0


@pytest.mark.gpu
def test_a_malformed_request_is_rejected_without_touching_the_gpu(live_server):
    import httpx

    response = httpx.post(
        f"{live_server}/v1/chat/completions",
        json={"model": MODEL, "messages": []},
        timeout=30,
    )
    assert response.status_code == 422
