# Suite sanity gate

T4.2 requires the frozen task suite to score ≥ 80 % against a strong hosted model
before any backend comparison is meaningful. A suite that a 70B model cannot pass
is measuring itself, not the serving stack underneath it.

`hosted-gate.json` is that gate run. It lives here rather than in the parent
directory on purpose: `bench/report.py` reads `bench/results/<backend>.json`, and
a gate result sitting at `results/hosted.json` would be picked up as the published
hosted anchor. The two are not the same measurement. The gate qualifies the suite
and may run anywhere; the anchor belongs to the final benchmark session and must
come from the same environment as the other backends.

| | |
|---|---|
| Result | **20/20 (100 %)** — 8/8 single, 8/8 two-tool, 4/4 multi |
| Model | `llama-3.3-70b-versatile` (Groq, `on_demand` free tier) |
| Calls/task | 3.0, with **0** parse retries |
| Tokens | 42,495 in / 1,986 out |
| Suite | unchanged — `agent/tasks.json`, `agent/corpus/` and `SYSTEM_PROMPT` are as frozen |

Run off the measurement box, so the `hardware` block in the JSON describes the
machine that drove the API, not a machine under test. No GPU is involved in a
hosted run; the field is recorded for provenance only.

## Why the first attempt scored 40 %

The free tier caps this model at 12,000 tokens per minute. The suite exhausts that
by task 9, after which every remaining task died on a 429 before its first call —
nine of them with zero LLM calls in under 0.2 s. Not one failure was a wrong
answer. The suite was never the problem, and editing tasks or the prompt in
response would have degraded a frozen artefact to work around someone's quota.

The fix is in the loop: a refused request is waited out and re-issued, and the
waiting is tracked separately and excluded from the latency and wall-clock the
run reports. A 429 costs no tokens and yields no completion, so it is not a
result about the backend. Local backends never emit one, so the path is inert for
flashstack and vLLM and the workload stays identical across all three.
