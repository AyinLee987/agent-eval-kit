# LLM-as-reranker on BEIR NFCorpus — results

Branch: `LLM_ReRanker`. Run with `python
benchmarks/rag_recall_beir/run_benchmark_llm_rerank.py` (needs
`corpus.json`/`queries.json`/`qrels.json` from `download_nfcorpus.py`, and
`DEEPSEEK_API_KEY` in the sibling repo's `.env`). Raw output in
`results_llm_rerank.json`; this file is the write-up. Read alongside
`RESULTS.md` — same corpus, same 323 queries, same four baseline
pipelines, plus one new arm.

## The question

`RESULTS.md`'s "Production takeaway" section recommends dropping the
sibling repo's default `HeuristicReranker` (lexical-overlap + a fixed
authority weight) — it measurably reduces MRR in both benchmarks with no
metric it clearly improves. Does asking the chat model itself to judge
relevance do any better? `agent/rag/rerank.py`'s `CallableReranker` adapter
made this a ~40-line addition (`llm_reranker.py`), no harness changes.

## Setup

- **Reranker**: `deepseek-chat` (DeepSeek-V3), temperature 0.0, one call per
  query. Given the query and its up-to-30 fused RRF candidates (each
  truncated to 300 chars), asked to return a JSON array of 0–10 relevance
  scores, one per candidate, in order.
- Everything else — corpus, queries, qrels, chunking, embeddings, BM25/dense
  retrievers, `candidate_limit=30` — identical to `RESULTS.md`.

## A real failure, and what fixed it: zero-shot vs. one-shot

The first version of the prompt was zero-shot (instructions + a one-line
format example: `Example for 3 passages: [7, 2, 9]`). Result: **133/323
queries (41%) came back unparseable** — not malformed JSON, but the model
literally echoing back the placeholder description instead of doing the
task:

```
[json array of 30 numbers]
```

Reproduced directly (not just inferred from the aggregate failure count):
resending the exact failing queries' real prompts got the same degenerate
response on repeat, ruling out a one-off fluke. `CallableReranker` +
`RAGPipeline`'s existing exception handling degrade gracefully when this
happens (falls back to plain RRF order for that one query,
`degraded_components`) — so the pipeline didn't crash, but the resulting
`llm_rerank` numbers from that run were an uninterpretable blend of ~59%
real LLM reranking and ~41% silent RRF fallback, not a clean measurement of
anything.

**Fix**: rewrote the prompt as one-shot — a complete worked example (a
5-passage query, fully scored: `[9, 0, 3, 8, 0]`) before the real task,
instead of a one-line format description. Verified against the exact two
queries that failed before (25-query sample, 0 failures, including both
originally-failing cases) before committing to a full rerun. Full-323
result: **0/323 parse failures.** (`llm_reranker.py`'s `RerankCache` is
keyed on the prompt template's own text alongside model+query+passages,
specifically so an edit like this can't silently mix answers scored under
the old wording into a "new prompt" run — old entries just become
unreachable dead weight in the cache file.)

The lesson generalizes beyond this one benchmark: a single one-line format
example does not reliably communicate "there is no shortcut here, you must
produce all N real numbers" to a model being asked to listwise-score
~30 items in one shot — a full worked example does. Worth defaulting to
one-shot for any future LLM-as-judge/scorer/reranker prompt in this
toolkit, not just this one.

## Results (n=323 queries, k=5, all five pipelines)

| Pipeline | MRR | Recall@5 | Recall@10 | nDCG@5 | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|
| BM25-only | 0.448 | 0.066 | 0.087 | 0.319 | 0.283 | 177 |
| Dense-only | 0.577 | 0.093 | 0.126 | 0.426 | 0.391 | 209 |
| Hybrid RRF | 0.567 | 0.099 | 0.136 | 0.430 | 0.393 | 377 |
| Hybrid + HeuristicReranker | 0.556 | 0.092 | 0.120 | 0.398 | 0.358 | 368 |
| **Hybrid + LLM rerank (DeepSeek)** | **0.590** | **0.113** | **0.153** | **0.482** | **0.447** | **2227** |

LLM rerank is the top scorer on **every single metric**, not just one —
and by a wider margin than any other pairwise comparison in either
`RESULTS.md` benchmark.

## Significance: strong on Recall/nDCG, not quite there on MRR

Paired bootstrap tests (10k resamples, same 323 queries every time),
LLM rerank vs. each of the three retrieval-quality baselines:

| Metric | vs. Hybrid RRF | vs. Hybrid+Heuristic | vs. Dense-only |
|---|---|---|---|
| MRR | +0.023, p=0.176 | +0.035, p=0.070 | +0.013, p=0.44 (not shown above, same order) |
| Recall@5 | +0.015, **p=0.021** | +0.022, **p=0.001** | +0.021, **p=0.000** |
| Recall@10 | +0.017, **p=0.001** | +0.033, **p=0.000** | +0.026, **p=0.000** |
| nDCG@5 | +0.052, **p=0.000** | +0.085, **p=0.000** | +0.057, **p=0.000** |
| nDCG@10 | +0.054, **p=0.000** | +0.088, **p=0.000** | +0.056, **p=0.000** |

Every Recall/nDCG comparison clears significance, against all three
baselines, most at p<0.001. MRR is the one metric where the visible
improvement (+0.013 to +0.035) doesn't clear the bar at n=323 — consistent
with `RESULTS.md`'s own observation that MRR is the noisiest of these
metrics on this corpus (a single first-hit swap moves it, unlike
Recall/nDCG which aggregate over each query's ~79 relevant chunks). Read
this as "LLM rerank measurably surfaces more of what's relevant, with the
one specific caveat that 'is the very first result exactly right' isn't
distinguishable from the baselines here" — not as a mixed or inconclusive
result.

## The real cost: latency

**2227ms/query — roughly 6–12x slower than every other pipeline** (177–
377ms/query, all local computation over cached embeddings). This is one
real LLM call per query, scoring up to 30 candidates in a single request;
it is the dominant cost, dwarfing the local BM25/RRF/heuristic-rerank work
entirely. `RESULTS.md`'s "Production takeaway" already flagged this
tradeoff in the abstract before this benchmark existed to put a number on
it: LLM rerank is not a hot-path default the way RRF fusion is — it's the
right call when the ~2 extra seconds per query is affordable (offline
batch scoring, a lower-QPS product, or a query class important enough to
justify it) and not when it isn't.

## Caveats

- **One model, one prompt design, one run.** `deepseek-chat` at
  temperature 0.0, this specific one-shot prompt, `max_passage_chars=300`,
  `candidate_limit=30` — a different model, a longer per-passage budget, or
  a different worked example could move these numbers in either direction.
  No comparison against a real cross-encoder (see `RESULTS.md`'s
  Production takeaway — still the standard lower-latency alternative,
  untested in this repo; no local cross-encoder model or hosted rerank API
  is currently wired in).
- **Cost isn't measured here, only latency.** 323 real chat-completion
  calls with ~30 candidates each in the prompt is a non-trivial token
  spend; no per-query cost figure is reported.
- **The zero-shot failure finding is itself a real result, not just a
  bug fixed en route** — a 41% degenerate-response rate on a "return only
  a JSON array" instruction, at this candidate count, on this model, is
  worth knowing on its own before shipping any listwise LLM-scoring prompt
  without a worked example.
