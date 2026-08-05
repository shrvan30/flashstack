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

ACTIONS = ("calculator", "doc_search", "mock_api", "final")

SYSTEM_PROMPT = """You are a precise research assistant with three tools.

Reply with ONLY a single JSON object and nothing else. No prose, no markdown, no
code fences. The object must have exactly these keys:

{"thought": "<one short sentence>", "action": "<calculator|doc_search|mock_api|final>", "action_input": "<string>"}

Tools:
- doc_search: search the Halden Systems document library. action_input is a search phrase.
- calculator: evaluate arithmetic. action_input is an expression such as "1240 * 12".
- mock_api: look up live records. action_input is "get_order(A-1001)" or "get_weather(Tromso)".
- final: give the answer. action_input is the answer itself, as short as possible.

Rules:
- Never guess a number or a policy. Look it up with doc_search first.
- Use calculator for every arithmetic step. Do not do mental arithmetic.
- When you have the answer, use action "final" with just the answer in action_input.
- Keep final answers minimal: a number, a name, or one short phrase.

Example 1
User: What does the Torvald T4 cost?
{"thought": "I need the price from the catalogue.", "action": "doc_search", "action_input": "Torvald T4 price"}
Observation: [1] 02-products.md ... The Torvald T4 is a pressure transducer. It costs 875 euro per unit ...
{"thought": "The catalogue gives the price directly.", "action": "final", "action_input": "875 euro"}

Example 2
User: What do 3 Vantage V1 units cost in total?
{"thought": "First find the unit price.", "action": "doc_search", "action_input": "Vantage V1 price per unit"}
Observation: [1] 02-products.md ... The Vantage V1 is a thermal imaging module. It costs 3,150 euro per unit ...
{"thought": "Now multiply the unit price by three.", "action": "calculator", "action_input": "3150 * 3"}
Observation: 9450
{"thought": "That is the total.", "action": "final", "action_input": "9450 euro"}
"""

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


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
    ) -> None:
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.stream = stream

    def _complete(self, messages: list[dict], result: AgentResult) -> str:
        """One LLM call, recording per-call metrics from whichever transport is used."""
        from bench.metrics import call_chat

        text, metrics = call_chat(
            self.client,
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            stream=self.stream,
        )
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

        result.wall_s = time.perf_counter() - started
        return result
