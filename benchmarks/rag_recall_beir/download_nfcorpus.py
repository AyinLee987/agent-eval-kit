"""One-time data-prep: fetch BEIR NFCorpus (corpus, test queries, qrels)
from HuggingFace and write them to static local JSON files.

NFCorpus (Boteva et al., 2016) is a published, non-self-authored retrieval
benchmark from the biomedical/nutrition domain: 3,633 documents, 323
relevance-judged test queries, graded relevance (0/1/2). This is the
"external, real public benchmark" complement to benchmarks/rag_recall's
synthetic corpus -- see this benchmark's RESULTS.md for what a much larger,
externally-sourced n changes about the confidence picture.

Uses the HuggingFace ``datasets-server`` REST API (JSON rows, paginated at
100/request) rather than the ``datasets`` library, so this script has no
dependency beyond the standard library -- consistent with the rest of this
toolkit. This is a one-time fetch: the resulting JSON files are checked into
the repo, so ``run_benchmark.py`` itself never needs network access to
HuggingFace, only to the embedding endpoint.

Usage (from the evaluation/ repo root):

    python benchmarks/rag_recall_beir/download_nfcorpus.py
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
PAGE_SIZE = 100
DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
QRELS_URL = "https://huggingface.co/datasets/BeIR/nfcorpus-qrels/resolve/main/test.tsv"


def _fetch_json(url: str, *, retries: int = 5) -> dict:
    """GET with retries -- datasets-server returns transient 502/503 while
    it lazily builds a dataset's parquet conversion on first access."""

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 502, 503, 504):
                raise
            wait = min(30, 3 * 2 ** attempt)
            print(f"  ({exc.code}, retrying in {wait}s...)")
            time.sleep(wait)
    raise last_error  # type: ignore[misc]


def fetch_all_rows(dataset: str, config: str, split: str) -> List[dict]:
    """Page through the datasets-server rows API and return every row's data."""

    rows: List[dict] = []
    offset = 0
    while True:
        url = (
            f"{DATASETS_SERVER}?dataset={dataset}&config={config}&split={split}"
            f"&offset={offset}&length={PAGE_SIZE}"
        )
        page = _fetch_json(url)
        page_rows = page["rows"]
        if not page_rows:
            break
        rows.extend(item["row"] for item in page_rows)
        offset += len(page_rows)
        if len(page_rows) < PAGE_SIZE:
            break
        time.sleep(1.0)  # polite pacing against the free datasets-server API
    return rows


def fetch_qrels() -> List[dict]:
    """query-id / corpus-id / score triples from the published test split."""

    with urllib.request.urlopen(QRELS_URL, timeout=30) as response:
        text = response.read().decode("utf-8")
    lines = [line for line in text.strip().splitlines() if line.strip()]
    header, *data_lines = lines
    assert header.split("\t") == ["query-id", "corpus-id", "score"], header
    qrels = []
    for line in data_lines:
        query_id, corpus_id, score = line.strip().split("\t")
        qrels.append({"query_id": query_id, "corpus_id": corpus_id, "score": int(score)})
    return qrels


def main() -> None:
    print("Fetching qrels (test split)...")
    qrels = fetch_qrels()
    judged_query_ids = {row["query_id"] for row in qrels}
    print(f"  {len(qrels)} judgments over {len(judged_query_ids)} judged queries.")

    print("Fetching corpus (this is ~37 paginated requests)...")
    corpus_rows = fetch_all_rows("BeIR/nfcorpus", "corpus", "corpus")
    corpus = [
        {"id": row["_id"], "title": row["title"], "text": row["text"]}
        for row in corpus_rows
    ]
    print(f"  {len(corpus)} documents.")

    print("Fetching queries and filtering to judged test queries...")
    query_rows = fetch_all_rows("BeIR/nfcorpus", "queries", "queries")
    queries = [
        {"id": row["_id"], "text": row["text"]}
        for row in query_rows
        if row["_id"] in judged_query_ids
    ]
    print(f"  {len(queries)} judged test queries (of {len(query_rows)} total).")

    missing = judged_query_ids - {q["id"] for q in queries}
    if missing:
        raise SystemExit(f"{len(missing)} judged query ids have no matching query text: {missing}")

    corpus_ids = {doc["id"] for doc in corpus}
    dangling = [row for row in qrels if row["corpus_id"] not in corpus_ids]
    if dangling:
        raise SystemExit(
            f"{len(dangling)} qrel rows reference a corpus id not present in the "
            "downloaded corpus -- the corpus fetch is incomplete."
        )

    for name, data in (("corpus.json", corpus), ("queries.json", queries), ("qrels.json", qrels)):
        path = HERE / name
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
