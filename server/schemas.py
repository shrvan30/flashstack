"""OpenAI-compatible request and response shapes, plus SSE chunk formatting.

Kept free of any engine import so the wire format can be tested on CPU with no
model loaded. The chunk builders are plain functions returning dicts for the same
reason: the exact bytes a client sees are worth asserting on directly.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SSE_DONE = "data: [DONE]\n\n"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=128, ge=1, le=2048)
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    seed: int | None = None

    @field_validator("messages")
    @classmethod
    def _last_message_is_not_from_the_assistant(cls, messages: list[ChatMessage]):
        # A trailing assistant turn would mean "continue this", which the chat
        # template's generation prompt contradicts. Reject rather than guess.
        if messages[-1].role == "assistant":
            raise ValueError("the final message must not be from the assistant")
        return messages


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "flashstack"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def make_usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def completion_response(
    request_id: str,
    model: str,
    content: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    created: int | None = None,
) -> dict[str, Any]:
    """The non-streaming body: one choice, with usage."""
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created if created is not None else int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": make_usage(prompt_tokens, completion_tokens),
    }


def chunk(
    request_id: str,
    model: str,
    created: int,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One `chat.completion.chunk` object.

    `usage`/`metrics` are only attached to the final chunk. The OpenAI client
    tolerates unknown top-level fields, so `metrics` is a safe place to report
    decode throughput — which cannot go in a header, because headers are sent
    before the first token and throughput is only known after the last.
    """
    body: dict[str, Any] = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        body["usage"] = usage
    if metrics is not None:
        body["metrics"] = metrics
    return body


def sse(payload: dict[str, Any]) -> str:
    """Serialise one chunk as an SSE `data:` event.

    Separator is a blank line, and `json.dumps` is given no indentation, because a
    newline inside the payload would terminate the event early and truncate the
    stream at the client.
    """
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
