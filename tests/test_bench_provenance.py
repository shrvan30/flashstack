"""Provenance must survive the whole journey: call -> task -> JSON -> report.

Testing `bench.metrics` alone is not enough. The failure mode this guards
against is a figure arriving in the results file, or in the published report,
with its source label dropped somewhere in between — at which point the number
looks authoritative and nothing says what it measures.
"""

from __future__ import annotations

import json

import pytest

from agent.loop import AgentResult
from bench.metrics import CallMetrics
from bench.report import build_report, sources_of
from bench.run import TaskOutcome, provenance


class Args:
    """The subset of the CLI namespace `summarise` reads."""

    def __init__(self, **kwargs) -> None:
        self.backend = "flashstack"
        self.model = "Qwen/Qwen2.5-0.5B-Instruct"
        self.base_url = "http://localhost:8000/v1"
        self.stream = True
        self.gpu_cost_per_hr = 0.35
        self.input_price_per_mtok = 0.0
        self.output_price_per_mtok = 0.0
        self.__dict__.update(kwargs)


def outcome(**kwargs) -> TaskOutcome:
    base = dict(
        id="t1",
        tier="single",
        correct=True,
        answer="42",
        expected="42",
        llm_calls=2,
        parse_retries=0,
        throttle_waits=0,
        throttle_s=0.0,
        steps=2,
        prompt_tokens=100,
        completion_tokens=20,
        wall_s=1.5,
    )
    base.update(kwargs)
    return TaskOutcome(**base)


def test_agent_result_keeps_values_and_labels_aligned():
    """A value is appended with its label or not at all."""
    result = AgentResult(task_id="t1", answer=None)
    metrics = CallMetrics(
        ttft_ms=120.0,
        ttft_source="client-stream",
        decode_tps=30.0,
        decode_tps_source="client-stream",
        server_ttft_ms=95.0,
        server_ttft_source="final-chunk",
    )
    # Mimic Agent._complete's recording without needing a live client.
    for value, source, values, sources in (
        (metrics.ttft_ms, metrics.ttft_source, result.ttft_ms, result.ttft_sources),
        (
            metrics.decode_tps,
            metrics.decode_tps_source,
            result.decode_tps,
            result.decode_tps_sources,
        ),
        (
            metrics.server_ttft_ms,
            metrics.server_ttft_source,
            result.server_ttft_ms,
            result.server_ttft_sources,
        ),
    ):
        if value is not None:
            values.append(value)
            sources.append(source)

    assert len(result.ttft_ms) == len(result.ttft_sources)
    assert len(result.decode_tps) == len(result.decode_tps_sources)
    assert len(result.server_ttft_ms) == len(result.server_ttft_sources)
    # No server decode throughput was reported, so neither list grew.
    assert result.server_decode_tps == [] and result.server_decode_tps_sources == []


def test_provenance_reports_a_clean_streamed_run():
    outcomes = [
        outcome(
            ttft_ms=[120.0, 130.0],
            ttft_sources=["client-stream", "client-stream"],
            decode_tps=[30.0, 31.0],
            decode_tps_sources=["client-stream", "client-stream"],
            server_ttft_ms=[95.0, 99.0],
            server_ttft_sources=["final-chunk", "final-chunk"],
        )
    ]
    record = provenance(outcomes, streamed=True)

    assert record["streamed"] is True
    assert record["published"]["ttft_ms"]["publishable"] is True
    assert record["published"]["ttft_ms"]["mixed"] is False
    assert record["published"]["decode_tps"]["publishable"] is True
    # The server figures are present, and separate.
    assert record["cross_check"]["server_ttft_ms"]["n"] == 2
    # percentile() is nearest-rank, so p50 of two values is the lower one.
    assert record["cross_check"]["server_ttft_ms"]["p50"] == pytest.approx(95.0)
    assert "never substituted" in record["rule"]


def test_provenance_flags_a_non_streamed_run_as_unpublishable():
    outcomes = [
        outcome(
            ttft_ms=[1200.0],
            ttft_sources=["client-wall-clock"],
            decode_tps=[16.0],
            decode_tps_sources=["client-wall-clock"],
            metric_notes=["non-streamed call: ... not a true time-to-first-token"],
        )
    ]
    record = provenance(outcomes, streamed=False)
    assert record["published"]["ttft_ms"]["publishable"] is False
    assert record["notes"]


def test_provenance_detects_a_column_mixing_two_meanings():
    """The exact accident the persisted labels exist to catch."""
    outcomes = [
        outcome(
            ttft_ms=[120.0],
            ttft_sources=["client-stream"],
            decode_tps=[30.0],
            decode_tps_sources=["client-stream"],
        ),
        outcome(
            id="t2",
            ttft_ms=[95.0],
            ttft_sources=["final-chunk"],
            decode_tps=[31.0],
            decode_tps_sources=["client-stream"],
        ),
    ]
    record = provenance(outcomes, streamed=True)
    assert record["published"]["ttft_ms"]["mixed"] is True
    assert record["published"]["ttft_ms"]["publishable"] is False
    assert record["published"]["decode_tps"]["mixed"] is False


def test_provenance_rejects_an_unlabelled_figure():
    outcomes = [outcome(ttft_ms=[120.0, 130.0], ttft_sources=["client-stream"])]
    with pytest.raises(ValueError, match="source label"):
        provenance(outcomes, streamed=True)


# -- and it must reach the report ----------------------------------------


def payload(**overrides) -> dict:
    base = {
        "backend": "flashstack",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "stream": True,
        "suite": {"corpus": "acme", "tasks": 20},
        "hardware": {"gpu": "RTX 3090", "driver": "580", "cuda": "13.0", "torch": "2.11"},
        "commits": {"flashstack": "abc1234", "flash_attention_cuda": "def5678"},
        "provenance": provenance(
            [
                outcome(
                    ttft_ms=[120.0],
                    ttft_sources=["client-stream"],
                    decode_tps=[30.0],
                    decode_tps_sources=["client-stream"],
                    server_ttft_ms=[95.0],
                    server_ttft_sources=["final-chunk"],
                )
            ],
            streamed=True,
        ),
        "results": {
            "success_rate": 90.0,
            "correct": 18,
            "tasks": 20,
            "llm_calls_per_task": 3.0,
            "parse_retries_total": 1,
            "ttft_ms_p50": 120.0,
            "decode_tps_mean": 30.0,
            "task_latency_s_p50": 1.5,
            "task_latency_s_p95": 2.5,
            "cost_per_task": 0.0001,
            "cost_basis": "wall-clock GPU time at 0.35/hour",
        },
        "by_tier": {"single": {"tasks": 8, "correct": 8, "llm_calls_per_task": 2.0}},
        "tasks": [],
    }
    base.update(overrides)
    return base


def test_report_prints_the_source_of_each_figure():
    text = build_report({"flashstack": payload()})
    assert "## Metric provenance" in text
    assert "client-stream" in text
    # The server figure appears as a cross-check, labelled as such.
    assert "final-chunk" in text
    assert "never the number in the results table" in text


def test_report_warns_when_a_column_mixes_sources():
    mixed = payload()
    mixed["provenance"]["published"]["ttft_ms"]["mixed"] = True
    mixed["provenance"]["published"]["ttft_ms"]["unique_sources"] = [
        "client-stream",
        "final-chunk",
    ]
    mixed["provenance"]["published"]["ttft_ms"]["publishable"] = False
    text = build_report({"flashstack": mixed})
    assert "Comparability warnings" in text
    assert "mixes" in text


def test_report_states_the_throttle_and_retry_accounting_rule():
    text = build_report({"flashstack": payload()})
    assert "Parse retries** are billed work" in text
    assert "Throttle waits** are not work" in text


def test_report_handles_results_written_before_provenance_existed():
    """Old result files must degrade loudly, not silently look authoritative."""
    legacy = payload()
    del legacy["provenance"]
    text = build_report({"flashstack": legacy})
    assert "predate provenance recording" in text
    assert sources_of(legacy, "ttft_ms") == "unlabelled"


def test_provenance_survives_a_json_round_trip():
    """The results file is the only thing a later report run gets to read."""
    original = payload()
    restored = json.loads(json.dumps(original))
    assert restored["provenance"] == original["provenance"]
    assert sources_of(restored, "ttft_ms") == "client-stream"
