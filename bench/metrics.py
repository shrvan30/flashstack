"""One LLM call, plus the per-call timing the comparison needs.

Where the numbers come from depends on the transport, and that asymmetry is
deliberate rather than incidental. flashstack reports time-to-first-token and
decode throughput on a **non-streamed** response as the `x-ttft-ms` and
`x-decode-tps` headers, because the whole generation is finished before the
response begins. On a **streamed** response it can still send a real `x-ttft-ms`
header — the server waits for the first token before the response starts — but
not `x-decode-tps`, since HTTP headers precede the body and decode throughput is
only known after the last token. Streaming therefore carries throughput and usage
in the final chunk's `metrics` field.

This module reads both paths, and falls back to client-side measurement for
backends that report neither (vLLM and any hosted endpoint). The `source` field
records which path a given measurement came from, so a report can never silently
mix server-reported and client-measured figures without saying so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

TTFT_HEADER = "x-ttft-ms"
DECODE_TPS_HEADER = "x-decode-tps"


@dataclass
class CallMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float | None = None
    decode_tps: float | None = None
    wall_ms: float = 0.0
    ttft_source: str = "unavailable"
    decode_tps_source: str = "unavailable"


def _to_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _extra(obj) -> dict:
    """Fields a server sent that the OpenAI schema does not define."""
    if obj is None:
        return {}
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra
    if isinstance(obj, dict):
        return obj
    return {}


def read_non_streamed(response, headers, wall_ms: float) -> CallMetrics:
    """Metrics for a non-streamed call: headers first, client clock as fallback."""
    metrics = CallMetrics(wall_ms=wall_ms)

    usage = getattr(response, "usage", None)
    if usage is not None:
        metrics.prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        metrics.completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    getter = getattr(headers, "get", None)
    ttft = _to_float(getter(TTFT_HEADER)) if getter else None
    decode_tps = _to_float(getter(DECODE_TPS_HEADER)) if getter else None

    if ttft is not None:
        metrics.ttft_ms, metrics.ttft_source = ttft, "header"
    else:
        # Without a header the whole call is the only observable, and a
        # non-streamed response gives no view of when the first token appeared.
        metrics.ttft_ms, metrics.ttft_source = wall_ms, "client-wall-clock"

    if decode_tps is not None:
        metrics.decode_tps, metrics.decode_tps_source = decode_tps, "header"
    elif metrics.completion_tokens > 1 and wall_ms > 0:
        metrics.decode_tps = metrics.completion_tokens / (wall_ms / 1e3)
        metrics.decode_tps_source = "client-wall-clock"

    return metrics


def read_streamed(chunks_seen: dict, wall_ms: float) -> CallMetrics:
    """Metrics for a streamed call: final-chunk `metrics`, then headers, then clock."""
    metrics = CallMetrics(wall_ms=wall_ms)

    usage = chunks_seen.get("usage")
    if usage is not None:
        metrics.prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        metrics.completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    if not metrics.completion_tokens:
        metrics.completion_tokens = chunks_seen.get("delta_count", 0)

    reported = chunks_seen.get("metrics") or {}
    server_ttft = _to_float(reported.get("ttft_ms"))
    server_tps = _to_float(reported.get("decode_tps"))
    header_ttft = _to_float((chunks_seen.get("headers") or {}).get(TTFT_HEADER))

    if server_ttft is not None:
        metrics.ttft_ms, metrics.ttft_source = server_ttft, "final-chunk"
    elif header_ttft is not None:
        metrics.ttft_ms, metrics.ttft_source = header_ttft, "header"
    elif chunks_seen.get("first_delta_ms"):
        metrics.ttft_ms = chunks_seen["first_delta_ms"]
        metrics.ttft_source = "client-stream"

    if server_tps is not None:
        metrics.decode_tps, metrics.decode_tps_source = server_tps, "final-chunk"
    else:
        # Client-side decode rate excludes the wait for the first token, which is
        # what makes it comparable with a server-reported figure rather than a
        # diluted end-to-end average.
        first = chunks_seen.get("first_delta_ms")
        after_first = metrics.completion_tokens - 1
        if first is not None and after_first > 0 and wall_ms > first:
            metrics.decode_tps = after_first / ((wall_ms - first) / 1e3)
            metrics.decode_tps_source = "client-stream"

    return metrics


def call_chat(
    client,
    model: str,
    messages: list[dict],
    max_tokens: int = 200,
    temperature: float = 0.0,
    seed: int | None = None,
    stream: bool = False,
) -> tuple[str, CallMetrics]:
    """Make one chat completion and return `(text, metrics)`."""
    request: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        request["seed"] = seed

    started = time.perf_counter()

    if not stream:
        raw = client.chat.completions.with_raw_response.create(**request)
        response = raw.parse()
        wall_ms = (time.perf_counter() - started) * 1e3
        text = response.choices[0].message.content or ""
        return text, read_non_streamed(response, raw.headers, wall_ms)

    raw = client.chat.completions.with_raw_response.create(stream=True, **request)
    headers = dict(raw.headers)
    stream_object = raw.parse()

    parts: list[str] = []
    seen: dict = {"headers": headers, "delta_count": 0}
    for chunk in stream_object:
        if chunk.choices and chunk.choices[0].delta.content:
            if "first_delta_ms" not in seen:
                seen["first_delta_ms"] = (time.perf_counter() - started) * 1e3
            parts.append(chunk.choices[0].delta.content)
            seen["delta_count"] += 1
        if getattr(chunk, "usage", None):
            seen["usage"] = chunk.usage
        extra = _extra(chunk).get("metrics")
        if isinstance(extra, dict):
            seen["metrics"] = extra

    wall_ms = (time.perf_counter() - started) * 1e3
    return "".join(parts), read_streamed(seen, wall_ms)
