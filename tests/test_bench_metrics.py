"""CPU tests for per-call metric extraction, including both flashstack paths.

Deviation #4 gives streamed and non-streamed responses different metric
transports, so the harness has two code paths and both have to be exercised.
A regression in either would not fail loudly — it would quietly fall back to a
client-side estimate and the report would compare a server-reported number
against a wall-clock one without saying so.
"""

from __future__ import annotations

import pytest

from bench.metrics import (
    DECODE_TPS_HEADER,
    TTFT_HEADER,
    CallMetrics,
    read_non_streamed,
    read_streamed,
)


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeResponse:
    def __init__(self, usage=None) -> None:
        self.usage = usage


# -- non-streamed: headers are the source of truth ------------------------


def test_non_streamed_reads_both_headers():
    metrics = read_non_streamed(
        FakeResponse(FakeUsage(36, 12)),
        {TTFT_HEADER: "131.66", DECODE_TPS_HEADER: "12.22"},
        wall_ms=1200.0,
    )
    assert metrics.prompt_tokens == 36
    assert metrics.completion_tokens == 12
    assert metrics.ttft_ms == pytest.approx(131.66)
    assert metrics.decode_tps == pytest.approx(12.22)
    assert metrics.ttft_source == "header"
    assert metrics.decode_tps_source == "header"


def test_non_streamed_falls_back_to_the_clock_without_headers():
    """vLLM and hosted endpoints send no flashstack headers."""
    metrics = read_non_streamed(FakeResponse(FakeUsage(10, 20)), {}, wall_ms=1000.0)
    assert metrics.ttft_source == "client-wall-clock"
    assert metrics.ttft_ms == pytest.approx(1000.0)
    assert metrics.decode_tps_source == "client-wall-clock"
    assert metrics.decode_tps == pytest.approx(20.0)


def test_non_streamed_ignores_unusable_header_values():
    for value in ("", "not-a-number", "0", "-5"):
        metrics = read_non_streamed(
            FakeResponse(FakeUsage(5, 5)),
            {TTFT_HEADER: value, DECODE_TPS_HEADER: value},
            wall_ms=500.0,
        )
        assert metrics.ttft_source == "client-wall-clock", value
        assert metrics.decode_tps_source == "client-wall-clock", value


def test_non_streamed_without_usage_reports_no_decode_rate():
    metrics = read_non_streamed(FakeResponse(None), {}, wall_ms=800.0)
    assert metrics.prompt_tokens == 0 and metrics.completion_tokens == 0
    assert metrics.decode_tps is None


# -- streamed: the final chunk is the source of truth ---------------------


def test_streamed_prefers_the_final_chunk_metrics():
    metrics = read_streamed(
        {
            "usage": FakeUsage(36, 24),
            "metrics": {"ttft_ms": 118.4, "decode_tps": 15.1},
            "headers": {TTFT_HEADER: "118.4"},
            "first_delta_ms": 500.0,
            "delta_count": 24,
        },
        wall_ms=2000.0,
    )
    assert metrics.ttft_ms == pytest.approx(118.4)
    assert metrics.decode_tps == pytest.approx(15.1)
    assert metrics.ttft_source == "final-chunk"
    assert metrics.decode_tps_source == "final-chunk"


def test_streamed_uses_the_ttft_header_when_the_final_chunk_lacks_metrics():
    """x-ttft-ms is a real header even on a stream; decode throughput cannot be."""
    metrics = read_streamed(
        {
            "usage": FakeUsage(10, 11),
            "headers": {TTFT_HEADER: "99.5"},
            "first_delta_ms": 300.0,
            "delta_count": 11,
        },
        wall_ms=1300.0,
    )
    assert metrics.ttft_ms == pytest.approx(99.5)
    assert metrics.ttft_source == "header"
    assert metrics.decode_tps_source == "client-stream"
    # 10 tokens after the first, over the 1000 ms that followed it.
    assert metrics.decode_tps == pytest.approx(10.0)


def test_streamed_falls_back_entirely_to_the_client_clock():
    metrics = read_streamed(
        {"headers": {}, "first_delta_ms": 200.0, "delta_count": 21}, wall_ms=1200.0
    )
    assert metrics.ttft_source == "client-stream"
    assert metrics.ttft_ms == pytest.approx(200.0)
    assert metrics.decode_tps_source == "client-stream"
    assert metrics.decode_tps == pytest.approx(20.0)


def test_streamed_decode_rate_excludes_the_wait_for_the_first_token():
    """Diluting decode rate with prefill would understate every backend."""
    fast_prefill = read_streamed(
        {"headers": {}, "first_delta_ms": 100.0, "delta_count": 11}, wall_ms=1100.0
    )
    slow_prefill = read_streamed(
        {"headers": {}, "first_delta_ms": 600.0, "delta_count": 11}, wall_ms=1600.0
    )
    # Both decoded 10 tokens in 1000 ms; only the prefill wait differed.
    assert fast_prefill.decode_tps == pytest.approx(slow_prefill.decode_tps)
    assert fast_prefill.decode_tps == pytest.approx(10.0)


def test_streamed_counts_deltas_when_usage_is_absent():
    metrics = read_streamed(
        {"headers": {}, "first_delta_ms": 50.0, "delta_count": 7}, wall_ms=1050.0
    )
    assert metrics.completion_tokens == 7


def test_streamed_single_token_reports_no_decode_rate():
    metrics = read_streamed(
        {"headers": {}, "first_delta_ms": 80.0, "delta_count": 1}, wall_ms=90.0
    )
    assert metrics.decode_tps is None


def test_metrics_default_to_unavailable_sources():
    metrics = CallMetrics()
    assert metrics.ttft_source == "unavailable"
    assert metrics.decode_tps_source == "unavailable"
    assert metrics.ttft_ms is None and metrics.decode_tps is None


# -- the two paths must agree when the server reports the same thing ------


def test_both_paths_agree_on_a_server_reported_measurement():
    """The transport must not change the number, only where it is read from."""
    non_streamed = read_non_streamed(
        FakeResponse(FakeUsage(36, 24)),
        {TTFT_HEADER: "118.40", DECODE_TPS_HEADER: "15.10"},
        wall_ms=2000.0,
    )
    streamed = read_streamed(
        {
            "usage": FakeUsage(36, 24),
            "metrics": {"ttft_ms": 118.40, "decode_tps": 15.10},
            "headers": {TTFT_HEADER: "118.40"},
            "first_delta_ms": 118.40,
            "delta_count": 24,
        },
        wall_ms=2000.0,
    )
    assert non_streamed.ttft_ms == pytest.approx(streamed.ttft_ms)
    assert non_streamed.decode_tps == pytest.approx(streamed.decode_tps)
    assert non_streamed.prompt_tokens == streamed.prompt_tokens
    assert non_streamed.completion_tokens == streamed.completion_tokens
    assert non_streamed.ttft_source == "header"
    assert streamed.ttft_source == "final-chunk"
