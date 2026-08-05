"""Static batching scheduler.

Requests queue; the loop takes the first one, waits up to `window_ms` for more,
and runs whatever it has as one batch of at most `max_batch`. The batch prefills
sequence by sequence, then decodes in lockstep until every member has finished,
with completed sequences dropping out of the batch as they go.

Why prefill is serial and decode is batched: the prefill kernel computes dense
causal attention over a single sequence length and cannot express a padded batch,
while the decode kernel takes per-sequence lengths and handles ragged batches
natively. That asymmetry is the kernel's, and the scheduler is shaped around it.

Why *static* batching: a sequence that finishes early frees its slot, but no
waiting request takes its place until the whole batch drains. Continuous batching
would admit new work every step. It is out of scope here, and the cost of not
having it is visible in the metrics — see the study notes.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

import torch

from engine.sampling import SamplingParams, StopState, make_generator, sample

logger = logging.getLogger("flashstack.scheduler")

_counter = itertools.count()


@dataclass
class GenerationRequest:
    """One in-flight request and everything the scheduler needs to serve it."""

    prompt_ids: torch.Tensor
    params: SamplingParams
    model: str
    request_id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    created: int = field(default_factory=lambda: int(time.time()))

    # Filled in as the request runs.
    tokens: asyncio.Queue = field(default_factory=asyncio.Queue)
    submitted_at: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    finished_at: float | None = None
    generated: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    error: BaseException | None = None

    @property
    def prompt_tokens(self) -> int:
        return int(self.prompt_ids.shape[0])

    @property
    def completion_tokens(self) -> int:
        return len(self.generated)

    @property
    def ttft_ms(self) -> float:
        if self.first_token_at is None:
            return 0.0
        return (self.first_token_at - self.submitted_at) * 1e3

    @property
    def decode_tps(self) -> float:
        """Tokens per second across the decode phase, excluding time to first token."""
        if self.first_token_at is None or self.finished_at is None:
            return 0.0
        elapsed = self.finished_at - self.first_token_at
        after_first = len(self.generated) - 1
        if elapsed <= 0 or after_first <= 0:
            return 0.0
        return after_first / elapsed


class Metrics:
    """Rolling aggregates over the last `window` completed requests."""

    def __init__(self, window: int = 256) -> None:
        self.window = window
        self._ttft: deque[float] = deque(maxlen=window)
        self._tps: deque[float] = deque(maxlen=window)
        self._batch_sizes: deque[int] = deque(maxlen=window)
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_batches = 0
        self.batched_batches = 0  # batches that actually held more than one request

    def record_request(self, request: GenerationRequest) -> None:
        self.total_requests += 1
        self.total_prompt_tokens += request.prompt_tokens
        self.total_completion_tokens += request.completion_tokens
        if request.ttft_ms > 0:
            self._ttft.append(request.ttft_ms)
        if request.decode_tps > 0:
            self._tps.append(request.decode_tps)

    def record_batch(self, size: int) -> None:
        self.total_batches += 1
        self._batch_sizes.append(size)
        if size > 1:
            self.batched_batches += 1

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return ordered[index]

    def snapshot(self) -> dict:
        ttft = list(self._ttft)
        tps = list(self._tps)
        sizes = list(self._batch_sizes)
        return {
            "window": self.window,
            "total_requests": self.total_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "ttft_ms": {
                "p50": self._percentile(ttft, 0.50),
                "p95": self._percentile(ttft, 0.95),
                "samples": len(ttft),
            },
            "decode_tps": {
                "mean": (sum(tps) / len(tps)) if tps else 0.0,
                "p50": self._percentile(tps, 0.50),
                "samples": len(tps),
            },
            "batching": {
                "total_batches": self.total_batches,
                "batches_with_multiple_requests": self.batched_batches,
                "mean_batch_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
                "max_batch_size": max(sizes) if sizes else 0,
            },
        }


class Scheduler:
    """Owns the runner and serialises all GPU work through one loop."""

    def __init__(
        self,
        runner,
        max_batch: int = 4,
        window_ms: float = 25.0,
        metrics_window: int = 256,
    ) -> None:
        self.runner = runner
        self.max_batch = min(max_batch, runner.cache.max_batch)
        self.window_s = window_ms / 1e3
        self.metrics = Metrics(metrics_window)
        self._queue: asyncio.Queue[GenerationRequest] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="flashstack-scheduler")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, request: GenerationRequest) -> GenerationRequest:
        await self._queue.put(request)
        return request

    # -- the loop ----------------------------------------------------------

    async def _run(self) -> None:
        while True:
            try:
                batch = await self._collect_batch()
                await self._serve_batch(batch)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - the loop must never die
                logger.exception("scheduler iteration failed")

    async def _collect_batch(self) -> list[GenerationRequest]:
        """Block for one request, then accept more for up to `window_s`."""
        first = await self._queue.get()
        batch = [first]
        deadline = asyncio.get_running_loop().time() + self.window_s

        while len(batch) < self.max_batch:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
            except (TimeoutError, asyncio.TimeoutError):
                break
        return batch

    async def _serve_batch(self, batch: list[GenerationRequest]) -> None:
        self.metrics.record_batch(len(batch))
        logger.info(
            "decode batch formed: %d request(s) [%s]",
            len(batch),
            ", ".join(r.request_id[-8:] for r in batch),
        )

        slots: dict[int, int] = {}
        states: dict[int, StopState] = {}
        generators: dict[int, torch.Generator | None] = {}
        pending: list[GenerationRequest] = []

        try:
            for request in batch:
                key = id(request)
                try:
                    slot = self.runner.allocate()
                except Exception as exc:
                    self._fail(request, exc)
                    continue
                slots[key] = slot
                states[key] = StopState(request.params)
                generators[key] = make_generator(request.params.seed)

                # Prefill is per sequence: the kernel has no padded-batch form.
                logits = self.runner.prefill(request.prompt_ids, slot)
                token = int(
                    sample(
                        logits,
                        request.params,
                        previous_tokens=request.prompt_ids,
                        generator=generators[key],
                    )
                )
                if self._emit(request, token, states[key]):
                    self._finish(request)
                    self.runner.free(slot)
                    del slots[key]
                else:
                    pending.append(request)
                await asyncio.sleep(0)  # let the event loop flush this token

            # Lockstep decode. Finished sequences drop out; the rest continue.
            while pending:
                active = pending
                active_slots = [slots[id(r)] for r in active]
                last_tokens = torch.tensor(
                    [r.generated[-1] for r in active], dtype=torch.long
                )
                logits = self.runner.decode_step(last_tokens, active_slots)

                still_running: list[GenerationRequest] = []
                for row, request in enumerate(active):
                    key = id(request)
                    token = int(
                        sample(
                            logits[row],
                            request.params,
                            previous_tokens=torch.tensor(request.generated),
                            generator=generators[key],
                        )
                    )
                    if self._emit(request, token, states[key]):
                        self._finish(request)
                        self.runner.free(slots.pop(key))
                    else:
                        still_running.append(request)

                pending = still_running
                await asyncio.sleep(0)
        finally:
            for slot in slots.values():
                try:
                    self.runner.free(slot)
                except Exception:  # pragma: no cover
                    logger.exception("failed to free slot %s", slot)
            for request in batch:
                if request.finished_at is None and request.error is None:
                    self._fail(request, RuntimeError("request aborted"))

    # -- per-request bookkeeping -------------------------------------------

    def _emit(self, request: GenerationRequest, token: int, state: StopState) -> bool:
        """Record and publish one token. Returns True when the request is done."""
        if request.first_token_at is None:
            request.first_token_at = time.perf_counter()

        finished = state.observe(token)
        # A stop token ends the sequence but is not part of the text.
        if state.finish_reason != "stop":
            request.generated.append(token)
            request.tokens.put_nowait(token)
        if finished:
            request.finish_reason = state.finish_reason
        return finished

    def _finish(self, request: GenerationRequest) -> None:
        request.finished_at = time.perf_counter()
        self.metrics.record_request(request)
        request.tokens.put_nowait(None)

    def _fail(self, request: GenerationRequest, exc: BaseException) -> None:
        request.error = exc
        request.finish_reason = request.finish_reason or "error"
        request.finished_at = time.perf_counter()
        request.tokens.put_nowait(None)
