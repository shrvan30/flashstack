"""Turn an nsys decode trace into a GPU busy-versus-idle figure.

`nsys stats --report cuda_gpu_trace` gives one row per GPU operation with a start
timestamp and a duration. Busy time is the measure of the **union** of those
intervals, not their sum: operations on different streams can overlap, and adding
them would let busy time exceed the wall clock. The gaps between the merged
intervals are what the GPU spent waiting for the host to hand it the next launch.

The output is a single percentage plus the per-step arithmetic behind it, written
as markdown so it can be cited directly by the backend comparison.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path


def gpu_rows(nsys: str, report: Path) -> list[dict]:
    """Run `nsys stats` and return the CUDA GPU trace rows."""
    result = subprocess.run(
        [nsys, "stats", "--report", "cuda_gpu_trace", "--format", "csv", str(report)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"nsys stats failed on {report}")

    # nsys prints progress lines before the CSV; the header row starts it.
    lines = result.stdout.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Start (ns)"):
            body = "\n".join(lines[index:])
            break
    else:
        raise SystemExit("no cuda_gpu_trace table in the nsys output")

    return list(csv.DictReader(io.StringIO(body)))


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of possibly-overlapping [start, end) intervals."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def analyse(rows: list[dict], steps: int) -> dict:
    intervals: list[tuple[int, int]] = []
    kernel_count = 0
    memory_count = 0
    for row in rows:
        try:
            start = int(row["Start (ns)"])
            duration = int(row["Duration (ns)"])
        except (KeyError, ValueError):
            continue
        intervals.append((start, start + duration))
        name = row.get("Name", "")
        if name.startswith("[CUDA mem"):
            memory_count += 1
        else:
            kernel_count += 1

    if not intervals:
        raise SystemExit("the trace contains no GPU operations")

    merged = merge_intervals(intervals)
    span_start = min(s for s, _ in merged)
    span_end = max(e for _, e in merged)
    span_ns = span_end - span_start
    busy_ns = sum(end - start for start, end in merged)
    naive_sum_ns = sum(end - start for start, end in intervals)

    gaps = []
    for index in range(1, len(merged)):
        gaps.append(merged[index][0] - merged[index - 1][1])
    gaps.sort(reverse=True)

    operations = kernel_count + memory_count
    return {
        "gpu_operations": operations,
        "kernels": kernel_count,
        "memory_ops": memory_count,
        "operations_per_step": operations / steps if steps else 0.0,
        "span_ms": span_ns / 1e6,
        "busy_ms": busy_ns / 1e6,
        "idle_ms": (span_ns - busy_ns) / 1e6,
        "busy_percent": 100.0 * busy_ns / span_ns if span_ns else 0.0,
        "idle_percent": 100.0 * (span_ns - busy_ns) / span_ns if span_ns else 0.0,
        "overlap_ratio": naive_sum_ns / busy_ns if busy_ns else 0.0,
        "mean_gap_us": (sum(gaps) / len(gaps) / 1e3) if gaps else 0.0,
        "largest_gaps_us": [g / 1e3 for g in gaps[:5]],
        "mean_op_us": busy_ns / operations / 1e3 if operations else 0.0,
    }


def render(stats: dict, wall: dict | None, report_name: str, baseline: dict | None) -> str:
    lines = [
        "# Decode dispatch overhead",
        "",
        "Where the wall clock goes during decoding, measured with Nsight Systems.",
        "This is a diagnosis, not a tuning exercise: nothing here was optimised as a",
        "result, and the figure exists so the backend comparison can attribute its",
        "gap to the right cause.",
        "",
        f"Trace: `{report_name}` — capture range opened after warm-up via",
        "`torch.cuda.profiler.start()`, so model load, prefill and context setup are",
        "excluded. GPU performance counters are unavailable on this host, so this is",
        "pure CUDA API/kernel tracing (`nsys -t cuda`, never `--gpu-metrics-device`).",
        "",
    ]

    if wall:
        lines += [
            "| run | |",
            "| :-- | --: |",
            f"| model | {wall['model']} |",
            f"| layers | {wall['layers']} |",
            f"| decode steps timed | {wall['steps']} |",
            f"| warm-up steps (untraced) | {wall['warmup']} |",
            f"| context length after run | {wall['context_after']} |",
            f"| wall clock | {wall['wall_s'] * 1e3:.1f} ms |",
            f"| per step | {wall['ms_per_step']:.2f} ms |",
            f"| throughput | {wall['tokens_per_s']:.1f} tok/s |",
            "",
        ]

    if baseline and wall:
        inflation = wall["ms_per_step"] / baseline["ms_per_step"]
        corrected_busy = 100.0 * (stats["busy_ms"] / wall["steps"]) / baseline["ms_per_step"]
        lines += [
            "### Tracing overhead, and the corrected figure",
            "",
            "Nsight Systems instruments every CUDA call, which lengthens exactly the",
            "host-side gaps this measurement is about. The same run was therefore",
            "timed without the profiler attached:",
            "",
            "| | ms/step | tok/s |",
            "| :-- | --: | --: |",
            f"| under nsys | {wall['ms_per_step']:.2f} | {wall['tokens_per_s']:.1f} |",
            f"| untraced | {baseline['ms_per_step']:.2f} | {baseline['tokens_per_s']:.1f} |",
            f"| inflation | {inflation:.2f}x | |",
            "",
            "GPU busy time per step is a property of the work, not of the observer, so",
            f"the {stats['busy_ms'] / wall['steps']:.2f} ms/step of measured GPU activity",
            "carries over unchanged. Against the untraced step time that gives:",
            "",
            f"- **GPU busy in normal operation: ~{corrected_busy:.0f}%**",
            f"- **GPU idle in normal operation: ~{100 - corrected_busy:.0f}%**",
            "",
            "The untraced figure is the one to quote. The traced percentages below",
            "are the raw measurement it is derived from.",
            "",
        ]

    lines += [
        "## GPU busy versus idle",
        "",
        "Busy time is the **union** of GPU operation intervals, not their sum:",
        "operations can overlap across streams, and summing would let busy time",
        "exceed the wall clock.",
        "",
        "| | |",
        "| :-- | --: |",
        f"| traced span | {stats['span_ms']:.2f} ms |",
        f"| **GPU busy** | **{stats['busy_ms']:.2f} ms ({stats['busy_percent']:.1f}%)** |",
        f"| **GPU idle** | **{stats['idle_ms']:.2f} ms ({stats['idle_percent']:.1f}%)** |",
        f"| GPU operations | {stats['gpu_operations']} ({stats['kernels']} kernels, "
        f"{stats['memory_ops']} memory) |",
        f"| operations per decode step | {stats['operations_per_step']:.0f} |",
        f"| mean operation duration | {stats['mean_op_us']:.1f} us |",
        f"| mean gap between operations | {stats['mean_gap_us']:.1f} us |",
        f"| largest gaps | {', '.join(f'{g:.0f} us' for g in stats['largest_gaps_us'])} |",
        "",
        "## Reading this",
        "",
        f"The GPU is idle {stats['idle_percent']:.0f}% of the traced wall clock. The engine",
        f"issues about {stats['operations_per_step']:.0f} GPU operations per token and each",
        f"one runs for {stats['mean_op_us']:.1f} us on average, separated by roughly",
        f"{stats['mean_gap_us']:.1f} us of nothing. Those gaps are the host: Python",
        "attribute lookups, tensor allocation, argument checking and the launch call",
        "itself, all of which happen between kernels rather than during them.",
        "",
        "The consequence for interpreting any backend comparison: decode throughput",
        "here is set by **host dispatch rate**, not by attention kernel speed. A",
        "faster attention kernel would shrink a small share of the busy fraction and",
        "leave the idle fraction untouched. The changes that would move this number",
        "are CUDA graphs (replay a captured launch sequence instead of re-issuing it),",
        "operator fusion, and continuous batching (more work per launch) — all of",
        "which are out of scope for this stage and none of which are about attention.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "nsys profile -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop \\",
        "    -o docs/profiles/decode_steps --force-overwrite true \\",
        "    python bench/profile_decode.py --steps 20 --summary /tmp/traced.json",
        "",
        "python bench/profile_decode.py --steps 20 --summary /tmp/untraced.json",
        "",
        "python bench/analyze_decode_trace.py \\",
        "    --report docs/profiles/decode_steps.nsys-rep \\",
        "    --wall-summary /tmp/traced.json --baseline-summary /tmp/untraced.json \\",
        "    --out docs/profiles/decode_dispatch.md",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--wall-summary", type=Path, default=None)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=None,
        help="wall-clock JSON from the same run without the profiler attached",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--nsys", default="nsys")
    args = parser.parse_args()

    wall = None
    if args.wall_summary and args.wall_summary.exists():
        wall = json.loads(args.wall_summary.read_text())

    baseline = None
    if args.baseline_summary and args.baseline_summary.exists():
        baseline = json.loads(args.baseline_summary.read_text())

    rows = gpu_rows(args.nsys, args.report)
    stats = analyse(rows, wall["steps"] if wall else 0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(stats, wall, args.report.name, baseline))
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
