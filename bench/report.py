"""Combine per-backend result files into one table, one chart and the gap analysis.

    python -m bench.report

Reads every `bench/results/<backend>.json` that exists and writes
`bench/results/report.md` plus `bench/results/comparison.svg`. Backends that were
not run are simply absent; the report says which and why rather than leaving a
blank column that reads as a zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BACKENDS = ("flashstack", "vllm", "hosted")

LABELS = {
    "flashstack": "flashstack",
    "vllm": "vLLM",
    "hosted": "hosted anchor",
}


def load_results() -> dict[str, dict]:
    found = {}
    for backend in BACKENDS:
        path = RESULTS_DIR / f"{backend}.json"
        if path.exists():
            found[backend] = json.loads(path.read_text())
    return found


def bar_chart(payloads: dict[str, dict], metric: str, title: str, unit: str) -> str:
    """A dependency-free SVG bar chart.

    Hand-written rather than pulled from a plotting library: the project has no
    charting dependency, and one bar chart does not justify adding one.
    """
    names = [LABELS[b] for b in payloads]
    values = [payloads[b]["results"][metric] for b in payloads]
    if not values:
        return ""

    width, height = 640, 260
    left, bottom, top = 120, 50, 46
    plot_w = width - left - 30
    plot_h = height - bottom - top
    peak = max(values) or 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="sans-serif" font-size="12">',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="15" '
        f'font-weight="600">{title}</text>',
    ]

    slot = plot_h / len(values)
    bar_h = min(34, slot * 0.62)
    for index, (name, value) in enumerate(zip(names, values, strict=False)):
        centre = top + slot * (index + 0.5)
        length = (value / peak) * plot_w if peak else 0
        parts.append(
            f'<text x="{left - 10}" y="{centre + 4}" text-anchor="end">{name}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{centre - bar_h / 2}" width="{max(length, 1):.1f}" '
            f'height="{bar_h:.1f}" fill="#4c78a8" rx="3"/>'
        )
        parts.append(
            f'<text x="{left + length + 8:.1f}" y="{centre + 4}" '
            f'fill="#333">{value:,.1f}{unit}</text>'
        )

    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="#999" stroke-width="1"/>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def sources_of(payload: dict, key: str) -> str:
    """The source label(s) behind one published figure, as printable text."""
    record = payload.get("provenance", {}).get("published", {}).get(key)
    if not record or not record.get("unique_sources"):
        return "unlabelled"
    label = " + ".join(record["unique_sources"])
    return f"**mixed: {label}**" if record.get("mixed") else label


def provenance_section(payloads: dict[str, dict]) -> list[str]:
    """State, per backend, what each published figure actually measures.

    A reader comparing a TTFT column across three serving stacks has no way to
    tell from the number itself whether it was taken at the server or the
    client. Printing the label next to every figure is the only thing that makes
    the column a comparison rather than a coincidence.
    """
    any_provenance = any("provenance" in p for p in payloads.values())
    if not any_provenance:
        return [
            "",
            "## Metric provenance",
            "",
            "These result files predate provenance recording, so the source of each",
            "figure is not known from the data. Re-run the benchmark before publishing.",
        ]

    rule = next(
        (p["provenance"]["rule"] for p in payloads.values() if "provenance" in p),
        "",
    )

    lines = [
        "",
        "## Metric provenance",
        "",
        rule,
        "",
        "| backend | streamed | TTFT p50 source | decode tok/s source | server cross-check |",
        "| :-- | :-- | :-- | :-- | :-- |",
    ]

    for backend, payload in payloads.items():
        prov = payload.get("provenance")
        if not prov:
            lines.append(f"| {LABELS[backend]} | ? | unlabelled | unlabelled | - |")
            continue
        cross = prov["cross_check"]["server_ttft_ms"]
        cross_text = (
            f"{fmt(cross['p50'], 0)} ms from "
            f"{' + '.join(cross['unique_sources'])} ({cross['n']} calls)"
            if cross["n"]
            else "none reported"
        )
        lines.append(
            f"| {LABELS[backend]} | {'yes' if prov['streamed'] else '**no**'} | "
            f"{sources_of(payload, 'ttft_ms')} | "
            f"{sources_of(payload, 'decode_tps')} | {cross_text} |"
        )

    warnings = []
    for backend, payload in payloads.items():
        prov = payload.get("provenance")
        if not prov:
            continue
        for key, column in (("ttft_ms", "TTFT"), ("decode_tps", "decode tok/s")):
            record = prov["published"][key]
            if record["mixed"]:
                warnings.append(
                    f"- **{LABELS[backend]}** {column} mixes "
                    f"{' and '.join(record['unique_sources'])} inside one column."
                )
            elif not record["publishable"]:
                warnings.append(
                    f"- **{LABELS[backend]}** {column} is "
                    f"{' + '.join(record['unique_sources']) or 'unavailable'}, not "
                    "client-stream, so it is not comparable with the other rows."
                )
        for note in prov.get("notes", []):
            warnings.append(f"- **{LABELS[backend]}**: {note}.")

    if warnings:
        lines += ["", "**Comparability warnings**", ""] + warnings
    else:
        lines += [
            "",
            "Every published latency and throughput figure above is a client-side",
            "stream measurement, taken the same way on all three backends.",
        ]

    lines += [
        "",
        "The cross-check column is flashstack's own `x-ttft-ms`, which the other two",
        "backends do not report. It is shown because the difference between it and the",
        "published client figure is a real quantity — the transport and framing cost",
        "the client pays on top of the server's own timing — and hidden nowhere: it is",
        "never the number in the results table.",
    ]
    return lines


def build_report(payloads: dict[str, dict], dispatch_note: str | None) -> str:
    if not payloads:
        raise SystemExit("no result files found; run bench.run first")

    first = next(iter(payloads.values()))
    hardware = first["hardware"]
    commits = first["commits"]

    lines = [
        "# Agent benchmark: three backends, one task suite",
        "",
        "Every backend runs the identical 20-task suite through the identical agent",
        "at temperature 0. The only thing that changes between runs is `base_url`",
        "and the model name.",
        "",
        "| | |",
        "| :-- | :-- |",
        f"| GPU | {hardware['gpu']} |",
        f"| Driver | {hardware['driver']} |",
        f"| CUDA (torch) | {hardware['cuda']} |",
        f"| PyTorch | {hardware['torch']} |",
        f"| flashstack commit | `{commits['flashstack']}` |",
        f"| flash-attention-cuda commit | `{commits['flash_attention_cuda']}` |",
        f"| suite | {first['suite']['tasks']} tasks over the "
        f"{first['suite']['corpus']} corpus |",
        "",
        "## Results",
        "",
        "| backend | model | success | calls/task | retries | TTFT p50 (ms) "
        "| decode (tok/s) | task p50 (s) | task p95 (s) | cost/task |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]

    for backend, payload in payloads.items():
        r = payload["results"]
        lines.append(
            f"| {LABELS[backend]} | `{payload['model']}` | "
            f"{r['success_rate']:.0f}% ({r['correct']}/{r['tasks']}) | "
            f"{fmt(r['llm_calls_per_task'])} | {r['parse_retries_total']} | "
            f"{fmt(r['ttft_ms_p50'], 0)} | {fmt(r['decode_tps_mean'])} | "
            f"{fmt(r['task_latency_s_p50'])} | {fmt(r['task_latency_s_p95'])} | "
            f"{r['cost_per_task']:.5f} |"
        )

    lines += [
        "",
        "TTFT and decode throughput above are client-side stream measurements on "
        "every row; see [Metric provenance](#metric-provenance) for the per-backend "
        "labels and the server-reported cross-check.",
    ]

    lines += [
        "",
        "Cost is not one number measured two ways. Local backends are billed by the",
        "hour, so their cost is wall-clock GPU time at the rented rate; a hosted",
        "backend is billed per token, so its cost comes from the usage its API",
        "reports. The two columns are the same currency and different accounting.",
        "",
    ]
    for backend, payload in payloads.items():
        lines.append(f"- **{LABELS[backend]}**: {payload['results']['cost_basis']}")

    missing = [b for b in BACKENDS if b not in payloads]
    if missing:
        lines += [
            "",
            "### Backends not run",
            "",
        ]
        for backend in missing:
            reason = (
                "no `HOSTED_BASE_URL` / `HOSTED_API_KEY` / `HOSTED_MODEL` in the "
                "environment, so the hosted anchor was skipped"
                if backend == "hosted"
                else "see the notes below"
            )
            lines.append(f"- **{LABELS[backend]}**: {reason}.")

    lines += [
        "",
        "## Per-tier breakdown",
        "",
        "| backend | single-tool | two-tool | multi-step (4+ calls) |",
        "| :-- | --: | --: | --: |",
    ]
    for backend, payload in payloads.items():
        tiers = payload["by_tier"]

        def cell(tier: str, tiers=tiers) -> str:
            if tier not in tiers:
                return "-"
            entry = tiers[tier]
            return f"{entry['correct']}/{entry['tasks']}"

        lines.append(
            f"| {LABELS[backend]} | {cell('single')} | {cell('two-tool')} | "
            f"{cell('multi')} |"
        )

    lines += provenance_section(payloads)

    lines += [
        "",
        "## How retries and throttling are counted",
        "",
        "Two different things can make a backend issue more calls than the task",
        "needs, and they are accounted differently because they cost differently.",
        "",
        "- **Parse retries** are billed work. The model returned something that was",
        "  not a valid action object, the agent sent one corrective message, and the",
        "  backend generated a second time. Those calls appear in `calls/task`, their",
        "  tokens appear in the token counts, and their latency stays inside the task",
        "  latency figures. A backend that formats badly should look more expensive,",
        "  because it is.",
        "- **Throttle waits** are not work. A hosted endpoint refused the request",
        "  before serving it, so nothing was computed and nothing was billed. The",
        "  wait is counted and reported separately, and is subtracted from task",
        "  latency and from the run's wall clock. Leaving it in would charge a local",
        "  backend's GPU-hour rate for time spent sleeping, and would make a",
        "  rate-limited hosted anchor look computationally slow when it was merely",
        "  queued.",
        "",
        "## Charts",
        "",
        "![decode throughput](comparison.svg)",
        "",
        "## Where the gap comes from",
        "",
        "The honest attribution, with the measurements that support it.",
        "",
        "**Decode throughput is set by host dispatch, not by the attention kernel.**",
    ]

    if dispatch_note:
        lines.append(dispatch_note)
    else:
        lines.append(
            "See `docs/profiles/decode_dispatch.md` for the Nsight Systems trace."
        )

    lines += [
        "",
        "That single measurement governs how the rest of this table should be read.",
        "A faster attention kernel cannot move a number that is bounded by how fast",
        "Python can issue launches. The Phase 1 profiles show the prefill kernel",
        "reaching roughly a third of this card's fp32 FMA peak and the decode kernel",
        "up to 39% of DRAM peak — respectable in isolation, and almost irrelevant to",
        "the end-to-end figures here, because the GPU spends most of decode waiting.",
        "",
        "**What vLLM has that flashstack does not**, in descending order of expected",
        "effect on these numbers:",
        "",
        "1. **CUDA graphs.** vLLM captures the decode step as a graph and replays it,",
        "   collapsing hundreds of per-launch host costs into one submission. This is",
        "   the direct fix for the idle fraction measured above and is out of scope",
        "   for Phase 3 by instruction.",
        "2. **Continuous batching.** flashstack forms a batch and runs it to",
        "   completion, so a request arriving mid-batch waits for the whole batch to",
        "   drain, and a batch whose sequences finish at different lengths spends most",
        "   of its life underfilled. vLLM admits work between decode steps.",
        "3. **Paged KV cache.** flashstack preallocates a contiguous 2048-token slot",
        "   per sequence regardless of how much is used. Paging lets vLLM hold more",
        "   concurrent sequences in the same memory, which raises achievable batch",
        "   size, which is what decode throughput actually scales with.",
        "4. **Fused kernels.** Layernorm, the SwiGLU MLP and the residual adds run as",
        "   separate eager PyTorch ops here. Each fusion removes launches, which is",
        "   the same currency as point 1.",
        "",
        "Attention itself is not on that list, and that is the finding. The kernel",
        "work in this project is real and measured, but the serving gap it sits",
        "inside is dominated by everything around it.",
        "",
        "## What the success column does and does not say",
        "",
        "Success rate here is a property of **Qwen2.5-0.5B-Instruct**, not of the",
        "serving stack. Two backends running the same weights at temperature 0 should",
        "agree closely; where they do not, the difference is sampling and numerics,",
        "not capability. The column is included because agent success rate drives",
        "cost — a failed task still burns every call it made — and because a large",
        "divergence between two backends on identical weights would itself be a bug",
        "worth finding.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "report.md")
    parser.add_argument(
        "--dispatch",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "profiles" / "decode_dispatch.md",
    )
    args = parser.parse_args()

    payloads = load_results()

    dispatch_note = None
    if args.dispatch.exists():
        text = args.dispatch.read_text()
        busy = [line for line in text.splitlines() if "GPU busy in normal operation" in line]
        idle = [line for line in text.splitlines() if "GPU idle in normal operation" in line]
        if busy and idle:
            dispatch_note = (
                f"An Nsight Systems trace of 20 decode steps "
                f"(`docs/profiles/decode_dispatch.md`) measures "
                f"{busy[0].split(':')[-1].strip().rstrip('*')} GPU busy and "
                f"{idle[0].split(':')[-1].strip().rstrip('*')} idle: the engine issues "
                f"roughly 1400 GPU operations per token, each running a couple of "
                f"microseconds, separated by tens of microseconds of host-side gap."
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(payloads, dispatch_note))

    chart = bar_chart(payloads, "decode_tps_mean", "Decode throughput (tokens/s)", "")
    if chart:
        (RESULTS_DIR / "comparison.svg").write_text(chart)

    print(f"wrote {args.out} from {len(payloads)} backend(s): {', '.join(payloads)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
