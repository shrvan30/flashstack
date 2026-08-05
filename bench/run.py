"""Run the task suite against one backend and record what it cost.

    python -m bench.run --backend flashstack --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-0.5B-Instruct --gpu-cost-per-hr 0.35

The backend is identified only by its `base_url` and model name; nothing in the
agent or the suite changes between runs. That is the whole design: the workload
is held fixed so the differences that show up are the serving stack's.

Cost is computed differently for local and hosted backends, because they bill
differently. A local backend is rented by the hour, so its cost is wall-clock
GPU-seconds times the hourly rate. A hosted backend bills tokens, so its cost
comes from the usage the API reports. Those two numbers are not the same kind of
thing and the report says so rather than putting them in one column unlabelled.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agent.loop import Agent
from bench.scoring import percentile, score

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = REPO_ROOT / "agent" / "tasks.json"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

# Published per-million-token prices for the hosted anchor, supplied by the
# operator because they change and are not discoverable from the API.
DEFAULT_INPUT_PRICE = 0.0
DEFAULT_OUTPUT_PRICE = 0.0


@dataclass
class TaskOutcome:
    id: str
    tier: str
    correct: bool
    answer: str | None
    expected: str
    llm_calls: int
    parse_retries: int
    steps: int
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    ttft_ms: list[float] = field(default_factory=list)
    decode_tps: list[float] = field(default_factory=list)
    error: str | None = None


def git_sha(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def hardware() -> dict:
    info = {"gpu": "unknown", "driver": "unknown", "cuda": "unknown", "torch": "unknown"}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        name, driver = result.stdout.strip().splitlines()[0].split(", ")
        info["gpu"], info["driver"] = name, driver
    except Exception:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "cpu"
    except Exception:
        pass
    return info


def load_tasks() -> tuple[list[dict], dict]:
    payload = json.loads(TASKS_PATH.read_text())
    return payload["tasks"], payload


def run_suite(args) -> dict:
    from openai import OpenAI

    tasks, suite = load_tasks()
    client = OpenAI(
        base_url=args.base_url,
        api_key=os.environ.get(args.api_key_env, "not-needed"),
        timeout=args.timeout,
        max_retries=0,
    )
    agent = Agent(
        client,
        model=args.model,
        max_steps=args.max_steps,
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stream=args.stream,
    )

    outcomes: list[TaskOutcome] = []
    started = time.perf_counter()

    for index, task in enumerate(tasks, start=1):
        result = agent.run(task["id"], task["prompt"])
        correct = score(result.answer, task["expected"], task.get("match", "contains"))
        outcomes.append(
            TaskOutcome(
                id=task["id"],
                tier=task["tier"],
                correct=correct,
                answer=result.answer,
                expected=task["expected"],
                llm_calls=result.llm_calls,
                parse_retries=result.parse_retries,
                steps=len(result.steps),
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                wall_s=result.wall_s,
                ttft_ms=result.ttft_ms,
                decode_tps=result.decode_tps,
                error=result.error,
            )
        )
        mark = "ok  " if correct else "FAIL"
        print(
            f"[{index:>2}/{len(tasks)}] {mark} {task['id']:<4} "
            f"{result.llm_calls} calls ({result.parse_retries} retries) "
            f"{result.wall_s:6.2f}s  answer={result.answer!r}"
        )

    total_wall = time.perf_counter() - started
    return summarise(args, suite, outcomes, total_wall)


def summarise(args, suite: dict, outcomes: list[TaskOutcome], total_wall: float) -> dict:
    count = len(outcomes)
    correct = sum(1 for o in outcomes if o.correct)
    latencies = [o.wall_s for o in outcomes]
    all_ttft = [t for o in outcomes for t in o.ttft_ms]
    all_tps = [t for o in outcomes for t in o.decode_tps]

    prompt_tokens = sum(o.prompt_tokens for o in outcomes)
    completion_tokens = sum(o.completion_tokens for o in outcomes)

    if args.backend == "hosted":
        cost_total = (
            prompt_tokens / 1e6 * args.input_price_per_mtok
            + completion_tokens / 1e6 * args.output_price_per_mtok
        )
        cost_basis = (
            f"token pricing ({args.input_price_per_mtok}/Mtok in, "
            f"{args.output_price_per_mtok}/Mtok out)"
        )
    else:
        cost_total = total_wall / 3600.0 * args.gpu_cost_per_hr
        cost_basis = f"wall-clock GPU time at {args.gpu_cost_per_hr}/hour"

    return {
        "backend": args.backend,
        "model": args.model,
        "base_url": args.base_url,
        "stream": args.stream,
        "suite": {"corpus": suite.get("corpus"), "tasks": count},
        "hardware": hardware(),
        "commits": {
            "flashstack": git_sha(REPO_ROOT),
            "flash_attention_cuda": git_sha(REPO_ROOT.parent / "flash-attention-cuda"),
        },
        "results": {
            "success_rate": 100.0 * correct / count if count else 0.0,
            "correct": correct,
            "tasks": count,
            "llm_calls_total": sum(o.llm_calls for o in outcomes),
            "llm_calls_per_task": sum(o.llm_calls for o in outcomes) / count if count else 0.0,
            "parse_retries_total": sum(o.parse_retries for o in outcomes),
            "ttft_ms_p50": percentile(all_ttft, 0.50),
            "ttft_ms_p95": percentile(all_ttft, 0.95),
            "decode_tps_mean": (sum(all_tps) / len(all_tps)) if all_tps else 0.0,
            "task_latency_s_p50": percentile(latencies, 0.50),
            "task_latency_s_p95": percentile(latencies, 0.95),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "wall_s_total": total_wall,
            "cost_total": cost_total,
            "cost_per_task": cost_total / count if count else 0.0,
            "cost_basis": cost_basis,
        },
        "by_tier": {
            tier: {
                "tasks": len([o for o in outcomes if o.tier == tier]),
                "correct": len([o for o in outcomes if o.tier == tier and o.correct]),
                "llm_calls_per_task": (
                    sum(o.llm_calls for o in outcomes if o.tier == tier)
                    / max(1, len([o for o in outcomes if o.tier == tier]))
                ),
            }
            for tier in sorted({o.tier for o in outcomes})
        },
        "tasks": [asdict(o) for o in outcomes],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["flashstack", "vllm", "hosted"])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-cost-per-hr", type=float, default=0.35)
    parser.add_argument("--input-price-per-mtok", type=float, default=DEFAULT_INPUT_PRICE)
    parser.add_argument("--output-price-per-mtok", type=float, default=DEFAULT_OUTPUT_PRICE)
    parser.add_argument("--api-key-env", default="HOSTED_API_KEY")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="drive the backend over SSE, exercising the final-chunk metrics path",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = run_suite(args)
    results = payload["results"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = args.out or RESULTS_DIR / f"{args.backend}.json"
    destination.write_text(json.dumps(payload, indent=2))

    print(
        f"\n{args.backend}: {results['correct']}/{results['tasks']} correct "
        f"({results['success_rate']:.0f}%), "
        f"{results['llm_calls_per_task']:.1f} calls/task, "
        f"TTFT p50 {results['ttft_ms_p50']:.0f} ms, "
        f"decode {results['decode_tps_mean']:.1f} tok/s, "
        f"latency p50 {results['task_latency_s_p50']:.1f}s, "
        f"cost/task {results['cost_per_task']:.5f}"
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
