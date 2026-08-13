# The OpenAI-compatible API server

The server's job: make the engine reachable by **any existing OpenAI client** — an
agent, the `openai` SDK, `curl` — by speaking the chat-completions protocol
exactly, including streaming. That compatibility is also what makes the benchmark
possible: the same harness drives this server, vLLM, and a hosted API by changing
nothing but `base_url`.

## Endpoints (`server/app.py`, schemas in `server/schemas.py`)

| Endpoint | Behaviour |
|---|---|
| `POST /v1/chat/completions` | Streamed (SSE, OpenAI chunk framing, final `[DONE]`) and non-streamed. Honours `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `stream`. |
| `GET /v1/models` | Lists the loaded models. |
| `GET /metrics` | Rolling aggregates (TTFT, decode tok/s, batch sizes) as JSON. |

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
for chunk in client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Explain gravity"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

## Static batching (`server/scheduler.py`)

Incoming requests queue; the scheduler groups up to **4** requests that arrive
within a **25 ms** window, runs one batched prefill, then **lockstep batched
decode** — every step advances the whole batch one token, and finished sequences
drop out of the batch while the rest continue.

The 25 ms window is a latency-for-throughput trade, and it is small on purpose:
under the sequential benchmark it costs nothing; under bursts it converts four
kernel launches into one. Batching is proven, not assumed — an end-to-end test
fires concurrent requests and asserts batch formation **from `/metrics`**, not
from a log line.

What static batching cannot do: admit a new request into a batch that is already
decoding. That is continuous batching, deliberately out of scope, and its absence
is priced in the benchmark report.

## Metrics, and two things HTTP forced

- **Honest TTFT.** The handler *awaits the first generated token* before sending
  response headers, so `x-ttft-ms` is a real time-to-first-token, not a
  time-to-headers.
- **Throughput cannot be a header on a stream.** Headers are sent before the body;
  decode tokens/s is only known after the last token. So: non-streamed responses
  carry both figures as headers; streamed responses carry the TTFT header and
  report throughput + usage in the **final chunk's metrics**. Discovered by
  building it, kept as a documented design decision.
- **Published numbers are client-side anyway.** The benchmark publishes TTFT and
  decode tok/s measured **client-stream on every backend** — the one meaning
  available on all three — and keeps the server-reported figures as labelled
  cross-checks. The gap between them is itself a finding: ~34.9 ms of SSE and
  serialization overhead (1.53x) on this server's path.

## Running it

```bash
python -m uvicorn server.app:app --port 8000
```

Requires the kernel extension installed (see the README's
[Reproduce](../README.md#15-reproduce) section) and a GPU with head-dim-64 model
weights available; first run downloads them from HuggingFace.

## Test coverage

CPU tests: request/response schema validation, SSE chunk framing (including the
final `[DONE]`), chat-template assembly, metrics bookkeeping — all with the kernel
mocked. GPU-marked tests: a full end-to-end streamed request through the real
engine, and the batching assertion above.
