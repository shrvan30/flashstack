"""One LLM call, plus the per-call timing the comparison needs.

Where the numbers come from depends on the transport, and that asymmetry is
deliberate rather than incidental. flashstack reports time-to-first-token and
decode throughput on a **non-streamed** response as the `x-ttft-ms` and
`x-decode-tps` headers, because the whole generation is finished before the
response begins. On a **streamed** response it can still send a real `x-ttft-ms`
header — the server waits for the first token before the response starts — but
not `x-decode-tps`, since HTTP headers precede the body and decode throughput is
only known after the last token. Streaming therefore carries throughput and usage
in the final chunk's `metrics` field. vLLM and hosted endpoints report neither,
and can only ever be measured from the client.

Only one of those three vantage points exists on all three backends, so that is
the one the published columns use:

**A published figure is always the client-side measurement.** A server-reported
figure is never promoted into a headline column, even when it is available and
even when it is the better measurement of the two. Preferring it where it exists
would mean the TTFT column held a server timestamp for flashstack and a client
timestamp for vLLM and the hosted anchor — three backends, two different
definitions, one column, no label. The comparison would then be partly measuring
the transport rather than the serving stack.

What a server does report is kept as a **cross-check**: `server_ttft_ms` and
`server_decode_tps` travel alongside the published pair, with their own source
labels. The difference between the two is itself informative — it is the client's
view of transport and framing overhead — but it is reported as a comparison, not
folded into the headline.

Every figure therefore carries the label of where it came from. Those labels are
persisted all the way into the results JSON and printed by the report, so no
column can silently mix two meanings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

TTFT_HEADER = "x-ttft-ms"
DECODE_TPS_HEADER = "x-decode-tps"

# Source labels. `client-stream` is the only one that is valid to publish, and
# only a streamed call can produce it.
CLIENT_STREAM = "client-stream"
CLIENT_WALL_CLOCK = "client-wall-clock"
HEADER = "header"
FINAL_CHUNK = "final-chunk"
UNAVAILABLE = "unavailable"

# The source a published latency/throughput column must carry for the comparison
# to be apples-to-apples across backends.
PUBLISHABLE_SOURCE = CLIENT_STREAM

# Why a non-streamed call cannot contribute a real TTFT: nothing is observable
# until the whole generation is finished, so the earliest client-side event is
# also the last one.
NON_STREAMED_TTFT_NOTE = (
    "non-streamed call: the client cannot observe the first token, so this is "
    "whole-call wall clock and is not a true time-to-first-token"
)


@dataclass
class CallMetrics:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Published pair — client-side, uniform across backends.
    ttft_ms: float | None = None
    decode_tps: float | None = None
    wall_ms: float = 0.0
    ttft_source: str = UNAVAILABLE
    decode_tps_source: str = UNAVAILABLE
    # Cross-check pair — whatever the server volunteered, never published.
    server_ttft_ms: float | None = None
    server_decode_tps: float | None = None
    server_ttft_source: str = UNAVAILABLE
    server_decode_tps_source: str = UNAVAILABLE
    notes: list[str] = field(default_factory=list)

    @property
    def ttft_is_publishable(self) -> bool:
        return self.ttft_source == PUBLISHABLE_SOURCE

    @property
    def decode_tps_is_publishable(self) -> bool:
        return self.decode_tps_source == PUBLISHABLE_SOURCE


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
    """Metrics for a non-streamed call.

    Nothing here is publishable. The client sees one event — the response
    arriving — so the best it can say about latency is the whole call, and the
    labels record exactly that rather than passing it off as a TTFT.
    """
    metrics = CallMetrics(wall_ms=wall_ms)

    usage = getattr(response, "usage", None)
    if usage is not None:
        metrics.prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        metrics.completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    getter = getattr(headers, "get", None)
    header_ttft = _to_float(getter(TTFT_HEADER)) if getter else None
    header_tps = _to_float(getter(DECODE_TPS_HEADER)) if getter else None

    metrics.ttft_ms, metrics.ttft_source = wall_ms, CLIENT_WALL_CLOCK
    metrics.notes.append(NON_STREAMED_TTFT_NOTE)

    if metrics.completion_tokens > 1 and wall_ms > 0:
        metrics.decode_tps = metrics.completion_tokens / (wall_ms / 1e3)
        metrics.decode_tps_source = CLIENT_WALL_CLOCK

    if header_ttft is not None:
        metrics.server_ttft_ms, metrics.server_ttft_source = header_ttft, HEADER
    if header_tps is not None:
        metrics.server_decode_tps, metrics.server_decode_tps_source = header_tps, HEADER

    return metrics


def read_streamed(chunks_seen: dict, wall_ms: float) -> CallMetrics:
    """Metrics for a streamed call.

    The published pair is measured at the client: TTFT is the arrival of the
    first content delta, decode rate is the tokens after it over the time after
    it. Whatever the server reported — final-chunk `metrics`, or the `x-ttft-ms`
    header — is recorded beside them as a cross-check and never substituted in.
    """
    metrics = CallMetrics(wall_ms=wall_ms)

    usage = chunks_seen.get("usage")
    if usage is not None:
        metrics.prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        metrics.completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    if not metrics.completion_tokens:
        metrics.completion_tokens = chunks_seen.get("delta_count", 0)

    first = _to_float(chunks_seen.get("first_delta_ms"))

    if first is not None:
        metrics.ttft_ms, metrics.ttft_source = first, CLIENT_STREAM

    # Client-side decode rate excludes the wait for the first token, which is
    # what makes it comparable with a server-reported figure rather than a
    # diluted end-to-end average.
    after_first = metrics.completion_tokens - 1
    if first is not None and after_first > 0 and wall_ms > first:
        metrics.decode_tps = after_first / ((wall_ms - first) / 1e3)
        metrics.decode_tps_source = CLIENT_STREAM

    reported = chunks_seen.get("metrics") or {}
    server_ttft = _to_float(reported.get("ttft_ms"))
    server_tps = _to_float(reported.get("decode_tps"))
    header_ttft = _to_float((chunks_seen.get("headers") or {}).get(TTFT_HEADER))

    if server_ttft is not None:
        metrics.server_ttft_ms, metrics.server_ttft_source = server_ttft, FINAL_CHUNK
    elif header_ttft is not None:
        metrics.server_ttft_ms, metrics.server_ttft_source = header_ttft, HEADER

    if server_tps is not None:
        metrics.server_decode_tps, metrics.server_decode_tps_source = server_tps, FINAL_CHUNK

    return metrics


def summarise_sources(values: list[float], sources: list[str]) -> dict:
    """Collapse index-aligned value/source lists into one provenance record.

    Raises if the two lists disagree in length: that would mean some figure in
    the run has no provenance, which is the exact failure this module exists to
    make impossible.
    """
    if len(values) != len(sources):
        raise ValueError(
            f"{len(values)} values carry {len(sources)} source labels; "
            "every published figure must have exactly one"
        )

    counts: dict[str, int] = {}
    for source in sources:
        counts[source] = counts.get(source, 0) + 1

    unique = sorted(counts)
    return {
        "n": len(values),
        "sources": counts,
        "unique_sources": unique,
        # A column mixing two source kinds is mixing two definitions. The report
        # must say so rather than print a single averaged number.
        "mixed": len(unique) > 1,
        "publishable": bool(unique) and unique == [PUBLISHABLE_SOURCE],
    }


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
