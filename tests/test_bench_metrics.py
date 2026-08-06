"""CPU tests for per-call metric extraction and the provenance that travels with it.

Deviation #4 gives streamed and non-streamed responses different metric
transports, so the harness has two code paths and both have to be exercised.
A regression in either would not fail loudly — it would quietly fall back to a
client-side estimate and the report would compare a server-reported number
against a wall-clock one without saying so.

The rule these tests pin down is that a **published** figure is always the
client-side measurement, on every backend, even when the server volunteered a
better one. The server's figure is kept beside it as a cross-check. Without
that rule the TTFT column would hold a server timestamp for flashstack and a
client timestamp for the other two backends, which is two definitions in one
column.
"""

from __future__ import annotations

import pytest

from bench.metrics import (
    DECODE_TPS_HEADER,
    PUBLISHABLE_SOURCE,
    TTFT_HEADER,
    CallMetrics,
    read_non_streamed,
    read_streamed,
    summarise_sources,
)


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeResponse:
    def __init__(self, usage=None) -> None:
        self.usage = usage


# -- non-streamed: nothing here is publishable ----------------------------


def test_non_streamed_never_publishes_a_header_as_ttft():
    """A header is a server timestamp; publishing it would break the column."""
    metrics = read_non_streamed(
        FakeResponse(FakeUsage(36, 12)),
        {TTFT_HEADER: "131.66", DECODE_TPS_HEADER: "12.22"},
        wall_ms=1200.0,
    )
    assert metrics.prompt_tokens == 36
    assert metrics.completion_tokens == 12
    # Published: whole-call wall clock, honestly labelled.
    assert metrics.ttft_ms == pytest.approx(1200.0)
    assert metrics.ttft_source == "client-wall-clock"
    assert not metrics.ttft_is_publishable
    # Cross-check: what the server said, kept but not promoted.
    assert metrics.server_ttft_ms == pytest.approx(131.66)
    assert metrics.server_ttft_source == "header"
    assert metrics.server_decode_tps == pytest.approx(12.22)
    assert metrics.server_decode_tps_source == "header"


def test_non_streamed_records_why_its_ttft_is_not_a_ttft():
    metrics = read_non_streamed(FakeResponse(FakeUsage(10, 20)), {}, wall_ms=1000.0)
    assert metrics.notes, "a non-streamed call must explain its TTFT"
    assert "not a true time-to-first-token" in metrics.notes[0]


def test_non_streamed_falls_back_to_the_clock_without_headers():
    """vLLM and hosted endpoints send no flashstack headers."""
    metrics = read_non_streamed(FakeResponse(FakeUsage(10, 20)), {}, wall_ms=1000.0)
    assert metrics.ttft_source == "client-wall-clock"
    assert metrics.ttft_ms == pytest.approx(1000.0)
    assert metrics.decode_tps_source == "client-wall-clock"
    assert metrics.decode_tps == pytest.approx(20.0)
    assert metrics.server_ttft_ms is None
    assert metrics.server_ttft_source == "unavailable"


def test_non_streamed_ignores_unusable_header_values():
    for value in ("", "not-a-number", "0", "-5"):
        metrics = read_non_streamed(
            FakeResponse(FakeUsage(5, 5)),
            {TTFT_HEADER: value, DECODE_TPS_HEADER: value},
            wall_ms=500.0,
        )
        assert metrics.server_ttft_ms is None, value
        assert metrics.server_ttft_source == "unavailable", value
        assert metrics.server_decode_tps is None, value


def test_non_streamed_without_usage_reports_no_decode_rate():
    metrics = read_non_streamed(FakeResponse(None), {}, wall_ms=800.0)
    assert metrics.prompt_tokens == 0 and metrics.completion_tokens == 0
    assert metrics.decode_tps is None


# -- streamed: the client measurement is the published one ----------------


def test_streamed_publishes_the_client_figure_and_keeps_the_server_one():
    """The whole point: server metrics present, and still not published."""
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
    assert metrics.ttft_ms == pytest.approx(500.0)
    assert metrics.ttft_source == PUBLISHABLE_SOURCE
    assert metrics.ttft_is_publishable
    # 23 tokens after the first, over the 1500 ms that followed it.
    assert metrics.decode_tps == pytest.approx(23 / 1.5)
    assert metrics.decode_tps_source == PUBLISHABLE_SOURCE

    assert metrics.server_ttft_ms == pytest.approx(118.4)
    assert metrics.server_ttft_source == "final-chunk"
    assert metrics.server_decode_tps == pytest.approx(15.1)
    assert metrics.server_decode_tps_source == "final-chunk"


def test_streamed_cross_check_falls_back_to_the_header():
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
    assert metrics.ttft_ms == pytest.approx(300.0)
    assert metrics.ttft_source == PUBLISHABLE_SOURCE
    assert metrics.server_ttft_ms == pytest.approx(99.5)
    assert metrics.server_ttft_source == "header"
    # No server throughput is possible on a stream without final-chunk metrics.
    assert metrics.server_decode_tps is None
    assert metrics.decode_tps == pytest.approx(10.0)


def test_streamed_without_any_server_metrics():
    metrics = read_streamed(
        {"headers": {}, "first_delta_ms": 200.0, "delta_count": 21}, wall_ms=1200.0
    )
    assert metrics.ttft_source == PUBLISHABLE_SOURCE
    assert metrics.ttft_ms == pytest.approx(200.0)
    assert metrics.decode_tps_source == PUBLISHABLE_SOURCE
    assert metrics.decode_tps == pytest.approx(20.0)
    assert metrics.server_ttft_ms is None and metrics.server_decode_tps is None


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
    assert metrics.server_ttft_source == "unavailable"
    assert metrics.server_decode_tps_source == "unavailable"
    assert metrics.ttft_ms is None and metrics.decode_tps is None
    assert not metrics.ttft_is_publishable


# -- the published figure must not depend on what the server volunteered --


def test_a_server_that_reports_nothing_yields_the_same_published_ttft():
    """flashstack and vLLM must be measured identically, not merely similarly.

    Same stream timings, one backend chatty about its own metrics and one
    silent. If the published TTFT differed between these, the column would be
    reporting the backend's reporting habits rather than its latency.
    """
    common = {"usage": FakeUsage(36, 24), "first_delta_ms": 412.0, "delta_count": 24}
    chatty = read_streamed(
        {
            **common,
            "metrics": {"ttft_ms": 118.4, "decode_tps": 15.1},
            "headers": {TTFT_HEADER: "118.4"},
        },
        wall_ms=2000.0,
    )
    silent = read_streamed({**common, "headers": {}}, wall_ms=2000.0)

    assert chatty.ttft_ms == pytest.approx(silent.ttft_ms)
    assert chatty.ttft_source == silent.ttft_source == PUBLISHABLE_SOURCE
    assert chatty.decode_tps == pytest.approx(silent.decode_tps)
    assert chatty.decode_tps_source == silent.decode_tps_source
    # Only the cross-check distinguishes them.
    assert chatty.server_ttft_ms is not None
    assert silent.server_ttft_ms is None


# -- provenance aggregation ------------------------------------------------


def test_summarise_sources_counts_each_label():
    record = summarise_sources(
        [1.0, 2.0, 3.0], ["client-stream", "client-stream", "header"]
    )
    assert record["n"] == 3
    assert record["sources"] == {"client-stream": 2, "header": 1}
    assert record["unique_sources"] == ["client-stream", "header"]
    assert record["mixed"] is True
    assert record["publishable"] is False


def test_summarise_sources_marks_a_uniform_client_stream_column_publishable():
    record = summarise_sources([1.0, 2.0], ["client-stream", "client-stream"])
    assert record["mixed"] is False
    assert record["publishable"] is True


def test_summarise_sources_rejects_a_figure_without_provenance():
    """The failure this whole mechanism exists to prevent."""
    with pytest.raises(ValueError, match="source label"):
        summarise_sources([1.0, 2.0], ["client-stream"])


def test_summarise_sources_handles_an_empty_run():
    record = summarise_sources([], [])
    assert record["n"] == 0
    assert record["mixed"] is False
    assert record["publishable"] is False
