"""A minimal ReAct loop over the OpenAI client.

This is a measurement workload, not a product. Its job is to generate realistic
multi-call traffic — a prefill and a decode for every step, with the context
growing as observations accumulate — so that three serving backends can be
compared on identical work. Anything that made the agent smarter but the traffic
less representative would be a regression here.

The backend is chosen entirely by `base_url` and model name. There is no provider
routing, no memory, no tool-choice logic beyond parsing what the model asked for.

Retries are counted and reported. A model that needs three attempts to emit valid
JSON has cost three prefills and three decodes, and that is a real difference
between backends serving models of different capability.

Throttle waits are a different thing and are counted separately. A hosted
endpoint on a shared tier can refuse a request outright with a 429; that costs no
tokens, produces no completion, and says nothing about the backend's speed or the
model's ability. Scoring it as a task failure would measure the account's quota
rather than the serving stack, so the loop waits and re-issues instead, and
subtracts the waiting from the latency it reports. Local backends never emit 429,
so this path is inert for them and the workload stays identical across all three.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from agent.tools.calculator import CalculatorError, calculate
from agent.tools.doc_search import search
from agent.tools.mock_api import call as mock_api_call

MAX_STEPS = 8
MAX_RETRIES = 3

# A refused request is re-issued rather than scored, but not forever: past this
# many consecutive refusals the quota is not going to clear within the run and
# the honest outcome is an error on the task. No single wait exceeds the length
# of a per-minute window, since that is the longest a token bucket can take to
# refill.
MAX_THROTTLE_RETRIES = 8
MAX_THROTTLE_WAIT_S = 60.0

# Roughly four characters per token. The conversation is trimmed to stay under
# this budget because the smallest backend in the comparison serves a 2048-token
# KV cache, and observations accumulate across up to eight steps. Overflowing it
# raises a 400 that would score as a task failure caused by the harness rather
# than by the backend under test. The same budget is applied to every backend so
# the workload stays identical.
MAX_CONTEXT_CHARS = 6000
ELIDED = "[earlier observation omitted to stay within the context window]"

ACTIONS = ("calculator", "doc_search", "mock_api", "final")

SYSTEM_PROMPT = """You answer questions using tools. Reply with ONE JSON object and nothing else.

{"thought": "<short>", "action": "<doc_search|mock_api|calculator|final>", "action_input": "<string>"}

- doc_search: Halden Systems prices, policies, company facts. Input: a search phrase.
- mock_api: orders and weather. Input: "get_order(A-1001)" or "get_weather(Tromso)".
- calculator: arithmetic. Input: "1240 * 12".
- final: the answer itself, as short as possible.

If the Observation already contains what you need, use "final" immediately. Never repeat a tool call you have already made.

Example 1
User: Which customer placed order A-1004?
{"thought": "An order ID means mock_api.", "action": "mock_api", "action_input": "get_order(A-1004)"}
Observation: {"order_id": "A-1004", "customer": "Gdansk Shipyard", "status": "cancelled"}
{"thought": "The record names the customer.", "action": "final", "action_input": "Gdansk Shipyard"}

Example 2
User: What do 3 Vantage V1 units cost in total?
{"thought": "Find the unit price.", "action": "doc_search", "action_input": "Vantage V1 price per unit"}
Observation: [1] 02-products.md ... The Vantage V1 is a thermal imaging module. It costs 3,150 euro per unit ...
{"thought": "Multiply it by three.", "action": "calculator", "action_input": "3150 * 3"}
Observation: 9450
{"thought": "That is the total.", "action": "final", "action_input": "9450 euro"}
"""

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

# Groq states the wait in the error body rather than a header; other providers
# use `retry-after`. Both are read, the header first.
_RETRY_HINT = re.compile(r"try again in ([0-9.]+)\s*(ms|s)\b", re.IGNORECASE)


@dataclass
class Step:
    action: str
    action_input: str
    observation: str
    thought: str = ""


@dataclass
class AgentResult:
    task_id: str
    answer: str | None
    steps: list[Step] = field(default_factory=list)
    llm_calls: int = 0
    parse_retries: int = 0
    throttle_waits: int = 0
    throttle_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: list[float] = field(default_factory=list)
    decode_tps: list[float] = field(default_factory=list)
    wall_s: float = 0.0
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.answer is not None


def extract_json(text: str) -> dict[str, Any]:
    """Recover the action object from a model response.

    Small instruct models wrap JSON in fences or add a sentence around it even
    when told not to. Tolerating that is not leniency about correctness — the
    object still has to contain a valid action — it just avoids scoring
    formatting habits as task failures, which would confound a comparison
    between backends running the same model.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    cleaned = _FENCE.sub("", text).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start == -1:
            raise ValueError(f"no JSON object found in {cleaned[:120]!r}") from None
        depth = 0
        for index in range(start, len(cleaned)):
            if cleaned[index] == "{":
                depth += 1
            elif cleaned[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(cleaned[start : index + 1])
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"malformed JSON object: {exc.msg}") from exc
                    break
        else:
            raise ValueError("JSON object is never closed")

    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")

    action = parsed.get("action")
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {list(ACTIONS)}, got {action!r}")

    action_input = parsed.get("action_input", "")
    if not isinstance(action_input, (str, int, float)):
        raise ValueError("action_input must be a string or a number")

    return {
        "thought": str(parsed.get("thought", "")),
        "action": action,
        "action_input": str(action_input),
    }


def trim_context(messages: list[dict], budget: int = MAX_CONTEXT_CHARS) -> list[dict]:
    """Elide the oldest observations until the conversation fits the budget.

    The system prompt and the original task are never dropped: without them the
    model has no instructions and no question. Observations are elided oldest
    first, and the two most recent are never elided — an early version trimmed
    the observation that held the looked-up price, and the model then had no way
    to answer and searched again until it hit the step limit.
    """
    total = sum(len(m["content"]) for m in messages)
    if total <= budget:
        return messages

    trimmed = [dict(m) for m in messages]
    protected = len(trimmed) - 4  # keep the last two observation exchanges intact
    for index in range(2, max(2, protected)):
        if total <= budget:
            break
        content = trimmed[index]["content"]
        if trimmed[index]["role"] == "user" and content.startswith("Observation:"):
            total -= len(content) - len(ELIDED)
            trimmed[index]["content"] = ELIDED
    return trimmed


def is_throttled(exc: Exception) -> bool:
    """Whether a failure was the backend refusing to serve, not failing to serve.

    Matched on the HTTP status rather than an SDK exception class so the agent
    keeps depending on nothing but a client with `.chat.completions`, which is
    what lets the same loop drive flashstack, vLLM and a hosted API unchanged.
    """
    return getattr(exc, "status_code", None) == 429


def throttle_delay(exc: Exception, attempt: int) -> float:
    """How long to wait before re-issuing a refused request.

    The server's own instruction is preferred, but it is a floor rather than the
    answer: a token-per-minute bucket tells each caller when *that* request would
    fit, which under sustained load is a few hundred milliseconds away and refuses
    again on arrival. Backing off exponentially underneath the server's figure
    guarantees the wait grows, so a run converges instead of spinning through its
    retry budget in a couple of seconds.
    """
    floor = min(2.0**attempt, MAX_THROTTLE_WAIT_S)

    hint = None
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            hint = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            hint = None

    if hint is None:
        match = _RETRY_HINT.search(str(exc))
        if match:
            hint = float(match.group(1))
            if match.group(2).lower() == "ms":
                hint /= 1e3

    if hint is None:
        return floor
    return min(max(hint + 0.25, floor), MAX_THROTTLE_WAIT_S)


def run_tool(action: str, action_input: str) -> str:
    if action == "doc_search":
        return search(action_input)
    if action == "calculator":
        try:
            return calculate(action_input)
        except CalculatorError as exc:
            return f"Calculator error: {exc}"
    if action == "mock_api":
        return mock_api_call(action_input)
    raise ValueError(f"not a tool action: {action}")


class Agent:
    """ReAct over an OpenAI-compatible endpoint. The backend is just a base_url."""

    def __init__(
        self,
        client,
        model: str,
        max_steps: int = MAX_STEPS,
        max_retries: int = MAX_RETRIES,
        temperature: float = 0.0,
        max_tokens: int = 200,
        seed: int | None = 0,
        stream: bool = False,
        max_throttle_retries: int = MAX_THROTTLE_RETRIES,
    ) -> None:
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.max_throttle_retries = max_throttle_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.stream = stream

    def _complete(self, messages: list[dict], result: AgentResult) -> str:
        """One LLM call, recording per-call metrics from whichever transport is used."""
        from bench.metrics import call_chat

        request = dict(
            model=self.model,
            messages=trim_context(messages),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            stream=self.stream,
        )

        # Only a served call is counted and timed. A refusal is retried outside
        # the measured region, so the metrics below describe the request that
        # actually ran and carry none of the waiting.
        for attempt in range(self.max_throttle_retries + 1):
            try:
                text, metrics = call_chat(self.client, **request)
                break
            except Exception as exc:
                if not is_throttled(exc) or attempt == self.max_throttle_retries:
                    raise
                wait = throttle_delay(exc, attempt)
                result.throttle_waits += 1
                result.throttle_s += wait
                time.sleep(wait)

        result.llm_calls += 1
        result.prompt_tokens += metrics.prompt_tokens
        result.completion_tokens += metrics.completion_tokens
        if metrics.ttft_ms:
            result.ttft_ms.append(metrics.ttft_ms)
        if metrics.decode_tps:
            result.decode_tps.append(metrics.decode_tps)
        return text

    def run(self, task_id: str, prompt: str) -> AgentResult:
        result = AgentResult(task_id=task_id, answer=None)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        started = time.perf_counter()

        try:
            for _ in range(self.max_steps):
                action = None
                for attempt in range(self.max_retries):
                    text = self._complete(messages, result)
                    try:
                        action = extract_json(text)
                        break
                    except ValueError as exc:
                        result.parse_retries += 1
                        if attempt == self.max_retries - 1:
                            break
                        messages.append({"role": "assistant", "content": text})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"That was not valid. {exc}. Reply with ONLY a JSON "
                                    'object: {"thought": "...", "action": "...", '
                                    '"action_input": "..."}'
                                ),
                            }
                        )

                if action is None:
                    result.error = "could not parse an action after retries"
                    break

                if action["action"] == "final":
                    result.answer = action["action_input"].strip()
                    result.steps.append(
                        Step("final", action["action_input"], "", action["thought"])
                    )
                    break

                observation = run_tool(action["action"], action["action_input"])
                result.steps.append(
                    Step(
                        action["action"],
                        action["action_input"],
                        observation,
                        action["thought"],
                    )
                )
                messages.append({"role": "assistant", "content": json.dumps(action)})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                result.error = f"hit the {self.max_steps}-step limit without answering"
        except Exception as exc:  # noqa: BLE001 - a backend failure is a result, not a crash
            result.error = f"{type(exc).__name__}: {exc}"

        # Time spent waiting out a quota is not time the backend took to do the
        # work, and leaving it in would make a throttled hosted run look slower
        # than the local ones for a reason that has nothing to do with serving.
        result.wall_s = time.perf_counter() - started - result.throttle_s
        return result
