"""BM25 retrieval over the corpus in `agent/corpus/`.

The corpus documents a company that does not exist, on purpose. If the questions
could be answered from the model's pretraining, the benchmark would measure
recall of the internet rather than the agent's ability to use a tool, and a
backend that never called `doc_search` would score the same as one that did.

Retrieval is deliberately unsophisticated. BM25 over whitespace tokens is enough
for a ten-document corpus, and anything cleverer would make the harness's own
quality a variable in a measurement about serving performance.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
MAX_RESULTS = 2
SNIPPET_CHARS = 1100

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def _index():
    """Load the corpus and build the BM25 index once per process."""
    from rank_bm25 import BM25Okapi

    paths = sorted(CORPUS_DIR.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"no corpus documents under {CORPUS_DIR}")

    documents = [path.read_text().strip() for path in paths]
    return paths, documents, BM25Okapi([tokenize(doc) for doc in documents])


def search(query: str, k: int = MAX_RESULTS) -> str:
    """Return the top-k documents as a readable block for the model to quote from.

    Whole documents are returned rather than sentence-level passages: a
    sentence-level snippet routinely strips the unit or currency from a number,
    which turns a retrieval task into a guessing task.

    Only two documents come back. The context window is a hard resource here — a
    0.5B model serving this suite has 2048 tokens of KV cache, and observations
    accumulate across up to eight steps; returning three full documents per search
    overflowed the cache by the fourth step, which scores as a task failure caused
    by the harness rather than by the backend.

    The per-document cap is set above the longest corpus document rather than
    tuned for context savings. An earlier 450-character cap truncated the product
    catalogue midway through, so the Torvald T4 and Vantage V1 prices were
    unreachable no matter how well a model searched — a harness defect that looks
    exactly like a model failure in the results table.
    """
    text = (query or "").strip()
    if not text:
        return "No query given. Provide search terms."

    paths, documents, bm25 = _index()
    scores = bm25.get_scores(tokenize(text))
    ranked = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:k]

    if not ranked or scores[ranked[0]] <= 0:
        return f"No documents matched {text!r}."

    blocks = []
    for rank, index in enumerate(ranked, start=1):
        if scores[index] <= 0:
            break
        body = documents[index]
        if len(body) > SNIPPET_CHARS:
            body = body[:SNIPPET_CHARS].rstrip() + " ..."
        blocks.append(f"[{rank}] {paths[index].name}\n{body}")

    return "\n\n".join(blocks)


def document_count() -> int:
    return len(_index()[0])
