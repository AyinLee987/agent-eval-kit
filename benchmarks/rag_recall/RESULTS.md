# RAG recall ablation — results

Run with `python benchmarks/rag_recall/run_benchmark.py`. Full per-pipeline,
per-style numbers are in `results.json`; this file is the write-up.

## Setup

- **Corpus**: 12 synthetic, general-domain documents (`corpus.py`) — a
  fictional company's engineering handbook (incident response, on-call,
  deployments, code review, data retention, security, expenses, VPN,
  backups, rate limits, leave, vendor review). 3 sections each, 36 child
  chunks total, ingested through the sibling project's
  `MedicalParentChildChunker` (works fine on non-medical Markdown — the
  medical-specific logic only protects specific Chinese clinical markers,
  which this corpus never triggers).
- **Queries**: 24 labeled queries (`queries.py`), 2 per document — one
  **lexical** (shares several words with the source sentence) and one
  **paraphrase** (asks the same thing in different words, deliberately with
  low word overlap). Ground truth is a fact substring resolved to exactly
  one chunk id at ingestion time (`run_benchmark.py:resolve_relevant_ids`),
  not a hand-picked chunk id.
- **Embedding**: a real endpoint — `qwen3.7-text-embedding` via an
  OpenAI-compatible DashScope-based endpoint (1024-dim), not the repo's
  zero-dependency hash embedding. `run_benchmark.py` refuses to run if
  `RAG_EMBEDDING_*` isn't configured, specifically so these numbers can't
  accidentally come from the hash fallback.
- **Pipelines compared**: `RAGPipeline` configured four ways —
  BM25-only, dense-only, hybrid (reciprocal rank fusion, no reranking), and
  hybrid + the repo's default `HeuristicReranker`.

## Results (n=24 queries, k=5)

| Pipeline               | MRR   | Recall@5 | Recall@10 | nDCG@10 |
|-------------------------|-------|----------|-----------|---------|
| BM25-only               | 0.648 | 0.750    | 0.792     | 0.680   |
| Dense-only              | 0.714 | 0.875    | 0.875     | 0.749   |
| **Hybrid (RRF)**        | **0.830** | **0.875** | **0.917** | **0.851** |
| Hybrid + HeuristicReranker | 0.711 | 0.833  | 0.917     | 0.760   |

**Hybrid RRF vs. BM25-only baseline: MRR +18.2pp (0.648→0.830), Recall@5
+12.5pp (0.750→0.875).**

### By query style (n=12 each), Recall@5

| Pipeline    | Lexical | Paraphrase |
|-------------|---------|------------|
| BM25-only   | 0.917   | 0.583      |
| Dense-only  | 0.917   | **0.833**  |
| Hybrid RRF  | 1.000   | 0.750      |
| Hybrid + Rerank | 0.917 | 0.750    |

All four pipelines are close to tied on lexical queries — BM25 alone is
already good when the query shares words with the source text, which is the
expected result and a useful sanity check that nothing here is broken.
**The gap is entirely in the paraphrase column**: BM25-only recovers only
58% of paraphrased-query answers in the top 5, while anything using the real
embedding recovers 75–83%. That gap is the actual evidence for "hybrid
retrieval helps" — it comes from genuine semantic matching, not from a
scoring-formula artifact, precisely because it only shows up on the queries
designed to have low word overlap with their answer.

## An honest surprise: the heuristic reranker sometimes *hurts*

Hybrid + `HeuristicReranker` scores *lower* on MRR than plain hybrid RRF
(0.711 vs 0.830) despite reranking running after fusion. The reranker's
formula (`rrf_score * 20 + lexical_overlap + authority`) pulls the ranking
back toward literal word overlap, which is exactly the signal paraphrased
queries lack — so on this benchmark it partially undoes the fusion's
semantic gain. This matches the sibling repo's own README, which calls
`HeuristicReranker` a "dependency-free baseline; replace with a validated
cross-encoder in production" — this benchmark is a concrete, measured
demonstration of *why* that matters, not just a caveat in prose.

## Caveats (read before citing a number from this on a resume)

- **n=24 queries / 36 chunks** is small — each query is worth ~4pp (~8pp
  within a 12-query style subset), so single-digit-pp differences are noise.
  The MRR gap between BM25-only and hybrid RRF (18pp) and the paraphrase
  Recall@5 gap (BM25 vs. dense-only, 25pp) are large enough to trust as
  directional; smaller gaps between the two hybrid variants are not.
- **One embedding model, one run.** No repeated sampling, no confidence
  interval, no second embedding model for comparison.
- **Synthetic corpus written to make the ablation legible** (clean
  one-section-one-chunk structure, deliberately paired lexical/paraphrase
  queries) — real-world recall numbers on messy source documents will differ.
- **What this benchmark is good evidence for**: hybrid retrieval measurably
  helps on queries that don't share vocabulary with their answer, and a
  term-overlap reranker can fight that benefit. **What it is not**: a
  claim about recall on any particular real-world corpus.
