# RAG recall on BEIR NFCorpus — results

Run with `python benchmarks/rag_recall_beir/download_nfcorpus.py` once, then
`python benchmarks/rag_recall_beir/run_benchmark.py`. Full numbers are in
`results.json`; this file is the write-up. `analyze_rrf_reordering.py`
digs into one specific finding below (see "A genuine surprise") against
real per-query retrieval data — free to rerun, only reads cached
embeddings.

This is the same four-pipeline ablation as `benchmarks/rag_recall/
RESULTS.md`, run against a published, non-self-authored benchmark instead
of a hand-written synthetic corpus — read alongside that file, not as a
replacement for it: the synthetic corpus is deliberately legible (clean
one-section-one-chunk structure, paired lexical/paraphrase queries designed
to isolate *why* hybrid retrieval helps); this one trades that legibility
for external validity and ~13x the query count.

## Setup

- **Corpus**: [NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)
  (Boteva et al., 2016) via the [BEIR](https://github.com/beir-cellar/beir)
  benchmark suite — 3,633 real biomedical/nutrition documents (abstracts and
  article excerpts), fetched from `BeIR/nfcorpus` on HuggingFace.
- **Queries**: 323 real test queries with published relevance judgments
  (`BeIR/nfcorpus-qrels`, test split), graded relevance (score 1 =
  partially relevant, 2 = highly relevant; both treated as relevant here —
  see caveats). Ground truth is the *published* qrels, not resolved from
  document text the way `benchmarks/rag_recall`'s fact-substring matching
  is.
- **Chunking**: same `MedicalParentChildChunker` as the synthetic
  benchmark, one `## title` section per document. Unlike the synthetic
  corpus's short hand-written sections, real NFCorpus abstracts often
  exceed the chunker's `max_tokens=400` and split into multiple child
  chunks — every chunk belonging to a relevant document counts as a
  correct retrieval (see `run_benchmark.py:build_cases`).
- **Embedding**: same real endpoint as `benchmarks/rag_recall` —
  `qwen3.7-text-embedding`, cached (`.embedding_cache.json`, not committed).
- **Pipelines compared**: identical to `benchmarks/rag_recall` — BM25-only,
  dense-only, hybrid (RRF, no reranking), and hybrid + the repo's default
  `HeuristicReranker`.
- **Statistics**: `agent_eval.stats.bootstrap_ci` / `paired_bootstrap_test`
  (10k resamples), same as the synthetic benchmark — see that file for the
  methodology note.

## Results (n=323 queries, k=5)

| Pipeline               | MRR   | MRR 95% CI (bootstrap) | Recall@5 | Recall@10 | nDCG@10 |
|-------------------------|-------|-------------------------|----------|-----------|---------|
| BM25-only               | 0.448 | [0.400, 0.496] | 0.066    | 0.087     | 0.283   |
| **Dense-only**          | **0.577** | [0.530, 0.624] | 0.093 | 0.126     | 0.391   |
| Hybrid (RRF)            | 0.567 | [0.519, 0.615] | **0.099** | **0.136** | **0.393** |
| Hybrid + HeuristicReranker | 0.556 | [0.507, 0.604] | 0.092 | 0.120     | 0.358   |

**Hybrid RRF vs. BM25-only: MRR +11.9pp (0.448→0.567).** Paired bootstrap
test (10k resamples, same 323 queries both sides): **diff=+0.119, p=0.000**
— the synthetic benchmark's headline finding (hybrid beats BM25-only)
replicates on a real, externally labeled corpus at 13x the query count.

Recall@k and nDCG here are far lower in absolute terms than
`benchmarks/rag_recall`'s (0.05–0.14 vs. 0.75–0.92) — expected, not a
regression: NFCorpus documents split into up to 20 child chunks each (see
Setup), so a query's relevant set is often several chunks scattered across
the corpus, and Recall@k requires finding *all* of them in the top-k. The
synthetic benchmark's ground truth was, by construction, exactly one chunk
per query — a much easier bar. MRR (rank of the *first* relevant hit) is
the metric comparable in spirit across both benchmarks.

### Is Recall@10 ≈ 0.09–0.14 actually low? Two checks, both say no

**1. The relevant set per query is large enough that even a perfect ranking
couldn't score much higher.** Computed directly from this run's own ground
truth: the 323 queries have an average of **79.4 relevant chunks each**
(median 36, max 997 — NFCorpus queries are broad topics with many relevant
passages, not single-fact lookups). A perfect ranking (every top-k slot
relevant) is capped at:

| k | Theoretical max Recall@k | Hybrid RRF (measured) | % of ceiling reached |
|---|--------------------------|------------------------|------------------------|
| 3 | 0.237 | 0.077 | 32.5% |
| 5 | 0.315 | 0.099 | 31.4% |
| 10 | 0.450 | 0.136 | 30.2% |

~30% of the achievable ceiling is a meaningful relative signal; the raw
0.136 alone is not — it looks close to zero only because the ceiling itself
is also far from 1.0, not because the pipeline is failing.

**2. Cross-checked against the published external baseline.** Pyserini's
official BM25 result on NFCorpus (document-level retrieval, from the BEIR
paper / reproducibility literature): **nDCG@10 ≈ 0.324, Recall@10 ≈ 0.159**.
This run's BM25-only: nDCG@10=0.283 (same order of magnitude — the gap is
explainable by this toolkit's binary-relevance nDCG treating qrel scores 1
and 2 identically, see caveats below), Recall@10=0.087 (visibly lower than
published). The Recall gap specifically traces to *granularity*, not
retrieval failure: the published baseline scores whole-document retrieval
(~38 relevant documents/query on average); this benchmark scores
chunk-level retrieval (~79 relevant chunks/query, see above) — a strictly
harder bar, since a document that's chunked into several pieces now
requires finding the *specific* relevant piece(s), not just the document.

## A genuine surprise: dense-only edges out hybrid RRF on MRR here

Unlike the synthetic benchmark (where hybrid RRF was the clear MRR winner),
**dense-only scores marginally higher MRR than hybrid RRF** (0.577 vs.
0.567) on NFCorpus, while hybrid RRF wins on every other metric (Recall@3/5/10,
nDCG@3/5/10). The 0.577 vs. 0.567 gap itself is small enough that its own
CIs overlap heavily ([0.530, 0.624] vs. [0.519, 0.615]); the honest read is
"roughly tied on MRR, hybrid ahead on Recall/nDCG," not "dense-only wins."

**What actually causes it** — verified against real per-query retrieval
data, not just theory (`analyze_rrf_reordering.py`, rerun any time at zero
cost since it only reads cached embeddings): sampled 5 queries where
dense-only's #1 hit was a real answer but hybrid RRF's #1 wasn't the same
chunk. Only **2 of the 5** actually cost hybrid its MRR point — the other 3
were hybrid promoting a *different, also-correct* chunk to #1, which is a
harmless reshuffle, not a loss. In both real losses, the same mechanism was
present, with the exact numbers to show it:

| Query | dense-only's #1 (correct) | BM25's rank for it | Hybrid's #1 instead | Its dense / BM25 ranks |
|---|---|---|---|---|
| *"What's Driving America's Obesity Problem?"* | chunk `8f1e89fa` | not in top 30 | chunk `e90e31c2` (**wrong**) | dense #9, BM25 #9 |
| *"Who Should be Careful About Curcumin?"* | chunk `cb9b3def` | not in top 30 | chunk `d5d0eb5e` (**wrong**) | dense #17, BM25 #4 |

`reciprocal_rank_fusion`'s score is `1/(60+rank)` per list a chunk appears
in, summed. Near the top of a list this barely changes with rank (rank 1 →
0.0164, rank 9 → 0.0145 — 12% less), so a chunk appearing in *two* lists at
a mediocre rank collects roughly double one list's max score. Concretely:

- Query 1: dense-only's correct pick scores `1/61 ≈ 0.0164` (only from
  dense — BM25 never nominated it as a candidate at all). The wrong chunk
  that displaces it scores `1/69 + 1/69 ≈ 0.0290` — being ranked #9 in
  *both* lists beats being #1 in just one.
- Query 2: same shape — `1/61 ≈ 0.0164` vs. `1/77 + 1/64 ≈ 0.0286`.

So the precise condition for hybrid losing to dense-only on a given query
is: **BM25 has zero keyword overlap with the correct chunk (can't even
nominate it into its own top-30), *and* some incorrect chunk happens to be
moderately plausible to both retrievers at once.** This is RRF behaving
exactly as designed — it structurally favors "two retrievers roughly agree"
over "one retriever is certain and the other has no opinion at all" — and
it's the same design choice that makes hybrid win on Recall/nDCG (pulling
in chunks either retriever alone would rank lower). It costs MRR
specifically only when the one retriever with an opinion happens to be
right and gets outvoted.

### The other direction: does BM25 ever actually save a query?

The above shows BM25 costing hybrid a correct #1. The symmetric question is
whether BM25 ever contributes a correct #1 that dense-only would have
missed on its own — checked the same way, against every query this time
(`analyze_bm25_contribution.py`), not just a 5-example sample:

Of 297 examinable queries, **17 (5.7%) had BM25 ranking a correct chunk
#1 while dense-only's own #1 was wrong** — i.e. dense-only alone would have
answered these 17 incorrectly. Hybrid RRF kept the correct answer in first
place on **15 of those 17 (88%)**. A few:

| Query | BM25's #1 | dense-only's #1 instead | Hybrid RRF's #1 |
|---|---|---|---|
| *"Avoiding Cooked Meat Carcinogens"* | correct (dense ranked it #2) | wrong | correct — kept BM25's pick |
| *"Finland"* | correct (dense ranked it #8) | wrong | correct — kept BM25's pick |
| *"DHA"* | correct (dense ranked it #14) | wrong | **wrong — lost it** |
| *"grapes"* | correct (dense ranked it #18) | wrong | **wrong — lost it** |

The pattern in the two "lost it" cases: BM25's correct pick was strong
lexically but ranked poorly by dense (#14, #18) — nowhere near enough to
accumulate a competitive combined RRF score against whatever BM25+dense
jointly preferred instead. Note "Finland", "DHA", "coffee", "grapes" are
one- or two-word queries with essentially no semantic ambiguity to resolve
— exactly where lexical matching is at its strongest and embedding
similarity has the least extra signal to add, which is consistent with
where BM25 earns its keep.

Put the two checks side by side: in the sampled cases, BM25 cost hybrid a
correct #1 roughly as often as it single-handedly saved one that dense
alone would have missed, and hybrid preserved the *correct* retriever's
call in the large majority of both directions (88% here; the
`analyze_rrf_reordering.py` sample was too small — 5 cases — to quote a
comparable percentage for the other direction without overstating it).
Fusion is not "BM25 mostly just gets in the way" — it is closer to a real,
two-way trade that nets out positive on every metric except MRR, where the
two effects roughly offset.

## The reranker's drop is no longer statistically distinguishable from noise

The synthetic benchmark found `HeuristicReranker` significantly *reducing*
MRR vs. plain hybrid RRF (p=0.007, n=24). Here, hybrid RRF → hybrid+rerank
still drops in the same direction (0.567→0.556, diff=−0.011) but the paired
test no longer clears significance: **p=0.262** at n=323 — a far larger
sample than the synthetic benchmark's 24 queries, yet the *same* comparison
that was significant there is not significant here. Read this as the
synthetic benchmark's own caveat ("smaller gaps between the two hybrid
variants are not [reliable]") validated directly, not contradicted: a small
true effect on real data can need more than 323 queries to separate from
noise, and "significant on the synthetic corpus" was never a promise it
would replicate at the same p-value on a different, harder, real one.

## Production takeaway: still hybrid RRF, without the reranker

Across both benchmarks, hybrid RRF wins Recall@k and nDCG@k over every
other pipeline, every time, at every k tested — the one place it does not
clearly win is MRR against dense-only, and that gap is a wash (CIs overlap;
the two analyses above show it's a real two-way trade, not one retriever
dragging the other down). For a RAG pipeline whose actual consumer is an
LLM reading the top `evidence_limit` chunks (15 here, not 1), Recall/nDCG —
how much of what's relevant made it into that whole window — is the more
representative metric than MRR, which only scores the single first hit.
BM25-only is not a safe substitute despite its low complexity: the
synthetic benchmark's paraphrase-query split showed it recovering only 58%
of paraphrased-query answers at k=5 versus 75–83% for anything with a real
embedding — production queries are typically *not* going to share
vocabulary with their source documents the way a lexical-style test query
does. Drop `HeuristicReranker`, though: it reduced MRR in both benchmarks
(significantly at n=24, directionally but not significantly at n=323) with
no metric it clearly improved — the sibling repo's own README already
calls it "a dependency-free baseline; replace with a validated
cross-encoder in production," and this is now measured evidence for that,
not just the caveat in prose. The one scenario where this recommendation
doesn't automatically hold: a product that only ever surfaces the single
top retrieved chunk to the user with no further reasoning step over the
rest of the evidence window — there, MRR is the metric that actually
matters, and dense-only's slight edge (itself within noise here) would be
worth a dedicated, larger-n test before deciding either way.

**Update (`LLM_ReRanker` branch): asking the chat model itself to rerank
beats `HeuristicReranker` outright, on every metric, most significantly —
see `RESULTS_llm_rerank.md`.** It doesn't replace the recommendation above
so much as sharpen it: use hybrid RRF (still the right retrieval-fusion
layer), but if a validated cross-encoder isn't available and the ~2
extra seconds/query is affordable for the product, an LLM-as-reranker step
on top measurably outperforms both plain RRF and the heuristic — with the
same MRR-vs-Recall/nDCG asymmetry as everywhere else in this benchmark.

## Caveats (read before citing a number from this on a resume)

- **Binary relevance from graded qrels.** NFCorpus's published judgments
  are graded (1/2); `agent_eval/retrieval_metrics.py`'s nDCG is
  binary-relevance only (1.0 if in the relevant set, 0.0 otherwise, per
  position) — this run treats score 1 and score 2 identically, which is
  *not* the official BEIR nDCG@10 protocol (that uses the graded scores
  directly via `pytrec_eval`). Numbers here are internally comparable
  across the four pipelines (same simplification applied to all of them)
  but should not be quoted as "the BEIR NFCorpus nDCG@10" figure from a
  paper or leaderboard.
- **One embedding model, one chunker/reranker configuration.** This
  measures the sibling repo's own `RAGPipeline` on NFCorpus, not a
  comparison against other retrieval systems' published NFCorpus numbers —
  different systems chunk, embed, and rerank differently, which materially
  changes recall even on identical ground truth.
- **The chunker was not tuned for this corpus.** `MedicalParentChildChunker`
  defaults (`target_tokens=250`, `max_tokens=400`) were carried over
  unchanged from the synthetic benchmark; NFCorpus's longer abstracts
  fragment more than the synthetic corpus's short sections did (see the
  chunking distribution reported by `run_benchmark.py`), which the
  document-level ground truth in `build_cases()` accounts for correctly,
  but a chunk-size sweep was not run.
- **What this benchmark is good evidence for**: whether the same
  conclusions from the synthetic ablation (hybrid retrieval beats BM25
  alone; the heuristic reranker can hurt) replicate on a real, externally
  labeled corpus at much larger n. **What it is not**: a claim that these
  numbers match any published NFCorpus leaderboard entry, or a tuned
  best-case result for this pipeline on this corpus.
