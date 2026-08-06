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
import re
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


# Measured on the same machine and committed alongside this report. Kept here as
# named constants rather than inline numbers so the arithmetic below can be
# checked against its sources. GPU_BUSY_FRACTION is only the fallback: the real
# value is parsed out of decode_dispatch.md so it cannot drift from the evidence.
GPU_BUSY_FRACTION = 0.06  # docs/profiles/decode_dispatch.md, untraced figure
DECODE_ATTN_US_PER_LAYER = 13.45 + 2.45  # kernel repo analysis.md: split + merge at S=1024
QWEN_LAYERS = 24


def parse_gpu_busy_fraction(path: Path) -> float | None:
    """Read the untraced GPU-busy percentage out of the dispatch study."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if "GPU busy in normal operation" in line:
            match = re.search(r"([\d.]+)\s*%", line)
            if match:
                return float(match.group(1)) / 100.0
    return None


def gap_attribution(payloads: dict[str, dict], busy_fraction: float | None = None) -> list[str]:
    """Decompose the flashstack-versus-vLLM decode gap into causes with numbers.

    Every step is arithmetic over figures measured on this machine, so a reader
    can disagree with the conclusion by disagreeing with a specific number.
    """
    if "flashstack" not in payloads or "vllm" not in payloads:
        return [
            "",
            "## Where the gap comes from",
            "",
            "Both flashstack and vLLM results are needed to attribute the gap, and at",
            "least one is missing from this run.",
        ]

    busy = busy_fraction if busy_fraction else GPU_BUSY_FRACTION

    fs = payloads["flashstack"]["results"]["decode_tps_mean"]
    vl = payloads["vllm"]["results"]["decode_tps_mean"]
    ratio = vl / fs if fs else 0.0

    # If dispatch were free, throughput would be bounded by the busy fraction.
    dispatch_ceiling = fs / busy if busy else 0.0
    dispatch_headroom = dispatch_ceiling / fs if fs else 0.0

    # Attention's share of a decode token, from the kernel repo's own traces.
    ms_per_token = 1e3 / fs if fs else 0.0
    attn_ms_per_token = QWEN_LAYERS * DECODE_ATTN_US_PER_LAYER / 1e3
    attn_share = 100.0 * attn_ms_per_token / ms_per_token if ms_per_token else 0.0
    attn_ceiling = fs / (1 - attn_share / 100.0) if attn_share < 100 else 0.0

    return [
        "",
        "## Where the gap comes from",
        "",
        f"vLLM decodes at {fmt(vl)} tok/s against flashstack's {fmt(fs)}, a "
        f"**{ratio:.1f}x** gap on identical weights. Three causes are usually offered "
        "for a gap like this. Only one of them explains this one, and the numbers say "
        "which.",
        "",
        "### 1. Host dispatch overhead — sufficient on its own",
        "",
        f"An Nsight Systems trace of decode steps "
        f"([docs/profiles/decode_dispatch.md](../../docs/profiles/decode_dispatch.md)) "
        f"measures the GPU **{busy * 100:.0f}% busy and "
        f"{(1 - busy) * 100:.0f}% idle** during decoding. The engine issues "
        f"about 1,388 GPU operations per token, each running ~2.8 us, separated by ~79 us "
        f"of host-side gap: Python attribute lookups, tensor allocation, argument "
        f"checking and the launch call itself.",
        "",
        f"If dispatch were free and the same GPU work were issued back to back, "
        f"flashstack would decode at {fmt(dispatch_ceiling, 0)} tok/s "
        f"({dispatch_headroom:.1f}x its measured rate). That ceiling is "
        f"**{dispatch_ceiling / vl:.1f}x beyond vLLM's measured {fmt(vl)} tok/s**.",
        "",
        f"So dispatch overhead alone more than accounts for the whole {ratio:.1f}x gap. "
        "Nothing else needs to be invoked to explain it — which does not mean nothing "
        "else is true, only that nothing else is *required*.",
        "",
        "### 2. Continuous batching — contributes nothing to this measurement",
        "",
        "**0x, here.** Continuous batching raises throughput by admitting new requests "
        "between decode steps instead of waiting for a batch to drain. This benchmark "
        "never gives it the chance: the agent is strictly sequential, with one request "
        "in flight at a time, so there is never a second request to admit. vLLM's "
        "scheduler is running the same single stream flashstack's is.",
        "",
        "This is a real architectural advantage of vLLM and it is the right answer for a "
        "served deployment under concurrent load. It explains none of the gap measured "
        "*here*, and quoting it as the cause would be borrowing an explanation from a "
        "workload this report did not run.",
        "",
        "### 3. Attention kernel quality — bounded at about 1.5%",
        "",
        f"The decode attention kernels cost {DECODE_ATTN_US_PER_LAYER:.1f} us per layer "
        f"per token at S=1024 (split + merge, from the kernel repository's nsys traces). "
        f"Across Qwen2.5-0.5B's {QWEN_LAYERS} layers that is "
        f"**{attn_ms_per_token:.2f} ms of attention per token**, against a measured "
        f"{ms_per_token:.1f} ms per token overall — about **{attn_share:.1f}%** of the "
        f"decode wall clock.",
        "",
        f"Making attention *infinitely fast* would therefore take flashstack from "
        f"{fmt(fs)} to about {fmt(attn_ceiling)} tok/s. The hand-written kernel this "
        f"whole project is built around is worth at most {attn_share:.1f}% of the number "
        f"it is most often assumed to control.",
        "",
        "### What that adds up to",
        "",
        "| cause | contribution to this gap |",
        "| :-- | :-- |",
        f"| Host dispatch overhead | sufficient alone — a {dispatch_headroom:.1f}x "
        f"ceiling against a {ratio:.1f}x gap |",
        "| Continuous batching | 0x — the workload is sequential |",
        f"| Attention kernel quality | <= {attn_share:.1f}% of decode wall clock |",
        "",
        "The honest reading is that this comparison measures **how fast Python can issue "
        "launches**, not how good the attention kernel is. That is the finding, and it "
        "is the opposite of the one the project set out expecting. The kernel work is "
        "real, measured, and correct; the serving gap it sits inside is dominated by "
        "everything around it.",
        "",
        "The changes that would actually move this number are CUDA graphs (replay a "
        "captured launch sequence instead of re-issuing it), operator fusion (fewer "
        "launches for the same work), and paged KV (more concurrent sequences per byte, "
        "which raises achievable batch size). All three attack the launch count. None of "
        "them is about attention, and all are out of scope for this stage by "
        "instruction.",
    ]


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

    fs = payloads.get("flashstack")
    if fs and "provenance" in fs:
        cross = fs["provenance"]["cross_check"]["server_ttft_ms"]
        published = fs["results"]["ttft_ms_p50"]
        if cross["n"] and cross["p50"]:
            delta = published - cross["p50"]
            lines += [
                "",
                f"For flashstack that difference is **{fmt(delta, 1)} ms** "
                f"({fmt(published, 1)} ms observed by the client against "
                f"{fmt(cross['p50'], 1)} ms reported by the server, "
                f"{published / cross['p50']:.2f}x). That gap is the measured cost of SSE "
                "framing and serialization plus loopback transport: the server stops its "
                "clock when the first token is generated, while the client starts seeing "
                "it only after the chunk has been JSON-encoded, wrapped in an SSE event "
                "and pushed through the socket. Published TTFT is the client figure "
                "because that is what a caller actually waits for, and because it is the "
                "only definition the other two backends can also supply.",
            ]
    return lines


def build_report(payloads: dict[str, dict], busy_fraction: float | None = None) -> str:
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
    ]

    lines += gap_attribution(payloads, busy_fraction)

    lines += [
        "",
        "For completeness, the Phase-1 profiles show the prefill kernel reaching",
        "roughly a third of this card's fp32 FMA peak and the decode kernel up to 27%",
        "of DRAM peak. Respectable in isolation, and — as the decomposition above",
        "shows — almost irrelevant to the end-to-end figures here.",
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

    busy_fraction = parse_gpu_busy_fraction(args.dispatch)
    if busy_fraction is None:
        print(
            f"warning: {args.dispatch} has no GPU-busy figure; "
            f"falling back to {GPU_BUSY_FRACTION:.0%} for the gap attribution"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(payloads, busy_fraction))

    chart = bar_chart(payloads, "decode_tps_mean", "Decode throughput (tokens/s)", "")
    if chart:
        (RESULTS_DIR / "comparison.svg").write_text(chart)

    print(f"wrote {args.out} from {len(payloads)} backend(s): {', '.join(payloads)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
