"""CPU tests for the server's wire format, request validation and scheduling maths.

No GPU, no model weights, no kernel. Everything here is either a pure function
over dicts and tensors or exercised against a stub runner, which is what lets the
server's externally-visible contract be regression-tested on a CI runner.
"""

from __future__ import annotations

import json

import pytest
import torch
from pydantic import ValidationError

from engine.sampling import SamplingParams
from server.app import _encode_prompt, _to_sampling_params
from server.scheduler import GenerationRequest, Metrics
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


class StubTokenizer:
    """Minimal tokenizer: one id per word, decode joins with spaces."""

    def __init__(self) -> None:
        self.seen_template_calls: list[tuple] = []

    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[abs(hash(w)) % 1000 for w in text.split()]])
        return type("Encoding", (), {"input_ids": ids})()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.seen_template_calls.append((tuple(m["role"] for m in messages), add_generation_prompt))
        rendered = " ".join(f"<{m['role']}>{m['content']}" for m in messages)
        return rendered + (" <assistant>" if add_generation_prompt else "")

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


class StubRunner:
    """A runner with no model behind it, for testing the server's own logic."""

    def __init__(self, with_chat_template: bool = True) -> None:
        self.tokenizer = StubTokenizer()
        self.device = torch.device("cpu")
        self.eos_token_ids = {99}
        if with_chat_template:
            self.apply_chat_template = self._apply_chat_template

    def _apply_chat_template(self, messages):
        text = self.tokenizer.apply_chat_template(messages)
        return self.tokenizer(text, return_tensors="pt").input_ids[0]


# -- request schema validation --------------------------------------------


def test_minimal_request_gets_deterministic_defaults():
    request = ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "hi"}]
    )
    assert request.temperature == 0.0
    assert request.top_p == 1.0
    assert request.max_tokens == 128
    assert request.stream is False
    assert request.seed is None


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": [{"role": "user", "content": "hi"}]},  # no model
        {"model": "m"},  # no messages
        {"model": "m", "messages": []},  # empty messages
        {"model": "m", "messages": [{"role": "bot", "content": "hi"}]},  # bad role
        {"model": "m", "messages": [{"role": "user"}]},  # no content
    ],
)
def test_malformed_requests_are_rejected(payload):
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", -0.1),
        ("temperature", 2.5),
        ("top_p", 0.0),
        ("top_p", 1.5),
        ("max_tokens", 0),
        ("max_tokens", 99999),
    ],
)
def test_out_of_range_sampling_fields_are_rejected(field, value):
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], field: value}
    with pytest.raises(ValidationError):
        ChatCompletionRequest(**payload)


def test_a_trailing_assistant_message_is_rejected():
    """It would contradict the generation prompt the template appends."""
    with pytest.raises(ValidationError, match="must not be from the assistant"):
        ChatCompletionRequest(
            model="m",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )


def test_system_then_user_is_accepted():
    request = ChatCompletionRequest(
        model="m",
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ],
    )
    assert [m.role for m in request.messages] == ["system", "user"]


# -- request -> sampling params -------------------------------------------


def test_request_maps_onto_sampling_params_with_the_model_s_eos():
    runner = StubRunner()
    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=32,
        seed=7,
    )
    params = _to_sampling_params(request, runner)
    assert isinstance(params, SamplingParams)
    assert (params.temperature, params.top_p, params.max_tokens, params.seed) == (
        0.7,
        0.9,
        32,
        7,
    )
    assert params.stop_token_ids == {99}
    assert not params.is_greedy


def test_default_request_produces_greedy_params():
    params = _to_sampling_params(
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}]),
        StubRunner(),
    )
    assert params.is_greedy


# -- chat template assembly -----------------------------------------------


def test_chat_template_is_used_when_the_runner_has_one():
    runner = StubRunner(with_chat_template=True)
    request = ChatCompletionRequest(
        model="m",
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ],
    )
    ids = _encode_prompt(runner, request)
    assert ids.dim() == 1 and ids.numel() > 0
    roles, add_generation_prompt = runner.tokenizer.seen_template_calls[-1]
    assert roles == ("system", "user")
    assert add_generation_prompt is True


def test_runner_without_a_template_falls_back_to_flattened_text():
    runner = StubRunner(with_chat_template=False)
    request = ChatCompletionRequest(
        model="gpt2", messages=[{"role": "user", "content": "hi there"}]
    )
    ids = _encode_prompt(runner, request)
    assert ids.dim() == 1 and ids.numel() > 0
    assert runner.tokenizer.seen_template_calls == []


# -- SSE and response bodies ----------------------------------------------


def test_sse_event_is_one_data_line_terminated_by_a_blank_line():
    event = sse({"a": 1})
    assert event.startswith("data: ")
    assert event.endswith("\n\n")
    assert event.count("\n\n") == 1


def test_sse_payload_contains_no_raw_newlines():
    """A newline inside the payload would end the event early and truncate it."""
    event = sse(chunk("id", "m", 0, {"content": "line one\nline two"}))
    body = event[len("data: ") : -2]
    assert "\n" not in body
    assert json.loads(body)["choices"][0]["delta"]["content"] == "line one\nline two"


def test_sse_is_compact_json():
    assert ", " not in sse({"a": 1, "b": 2})
    assert json.loads(sse({"a": 1, "b": 2})[6:-2]) == {"a": 1, "b": 2}


def test_chunk_has_the_openai_shape():
    body = chunk("chatcmpl-x", "m", 123, {"content": "hi"})
    assert body["object"] == "chat.completion.chunk"
    assert body["id"] == "chatcmpl-x"
    assert body["created"] == 123
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["delta"] == {"content": "hi"}
    assert choice["finish_reason"] is None
    assert "usage" not in body and "metrics" not in body


def test_final_chunk_carries_finish_reason_usage_and_metrics():
    body = chunk(
        "id",
        "m",
        0,
        delta={},
        finish_reason="stop",
        usage=make_usage(10, 5),
        metrics={"ttft_ms": 12.5, "decode_tps": 40.0},
    )
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    assert body["metrics"]["decode_tps"] == 40.0


def test_done_sentinel_is_the_literal_openai_terminator():
    assert SSE_DONE == "data: [DONE]\n\n"


def test_completion_response_shape():
    body = completion_response("id", "m", "hello", "stop", 4, 2, created=99)
    assert body["object"] == "chat.completion"
    assert body["created"] == 99
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 6


def test_model_list_shape():
    listing = ModelList(data=[ModelCard(id="my-model")])
    dumped = listing.model_dump()
    assert dumped["object"] == "list"
    assert dumped["data"][0]["id"] == "my-model"
    assert dumped["data"][0]["object"] == "model"


# -- request timing and metrics -------------------------------------------


def _finished_request(prompt: int, generated: int, ttft_s: float, decode_s: float):
    request = GenerationRequest(
        prompt_ids=torch.zeros(prompt, dtype=torch.long),
        params=SamplingParams(),
        model="m",
    )
    request.submitted_at = 0.0
    request.first_token_at = ttft_s
    request.finished_at = ttft_s + decode_s
    request.generated = list(range(generated))
    return request


def test_ttft_and_decode_tps_exclude_each_other():
    """Decode throughput must not be diluted by the prefill it waited on."""
    request = _finished_request(prompt=10, generated=11, ttft_s=0.5, decode_s=1.0)
    assert request.ttft_ms == pytest.approx(500.0)
    # 11 tokens generated, 10 of them after the first, over 1 second.
    assert request.decode_tps == pytest.approx(10.0)


def test_a_single_token_response_reports_no_decode_rate():
    request = _finished_request(prompt=5, generated=1, ttft_s=0.2, decode_s=0.0)
    assert request.decode_tps == 0.0
    assert request.ttft_ms == pytest.approx(200.0)


def test_unstarted_request_reports_zero_metrics():
    request = GenerationRequest(
        prompt_ids=torch.zeros(3, dtype=torch.long), params=SamplingParams(), model="m"
    )
    assert request.ttft_ms == 0.0 and request.decode_tps == 0.0
    assert request.prompt_tokens == 3 and request.completion_tokens == 0


def test_metrics_aggregate_over_requests_and_batches():
    metrics = Metrics(window=8)
    metrics.record_batch(1)
    metrics.record_batch(4)
    metrics.record_batch(2)
    for _ in range(3):
        metrics.record_request(_finished_request(10, 11, 0.1, 1.0))

    snapshot = metrics.snapshot()
    assert snapshot["total_requests"] == 3
    assert snapshot["total_prompt_tokens"] == 30
    assert snapshot["total_completion_tokens"] == 33
    assert snapshot["batching"]["total_batches"] == 3
    assert snapshot["batching"]["batches_with_multiple_requests"] == 2
    assert snapshot["batching"]["max_batch_size"] == 4
    assert snapshot["decode_tps"]["mean"] == pytest.approx(10.0)
    assert snapshot["ttft_ms"]["p50"] == pytest.approx(100.0)


def test_metrics_snapshot_is_empty_but_valid_before_any_traffic():
    snapshot = Metrics().snapshot()
    assert snapshot["total_requests"] == 0
    assert snapshot["ttft_ms"]["p50"] == 0.0
    assert snapshot["decode_tps"]["mean"] == 0.0
    assert snapshot["batching"]["max_batch_size"] == 0
    json.dumps(snapshot)  # must be serialisable as-is


def test_metrics_window_bounds_the_rolling_samples():
    metrics = Metrics(window=4)
    for _ in range(10):
        metrics.record_request(_finished_request(1, 2, 0.1, 1.0))
    snapshot = metrics.snapshot()
    assert snapshot["total_requests"] == 10, "totals are cumulative"
    assert snapshot["ttft_ms"]["samples"] == 4, "rolling stats are windowed"
