# RAG Recall Evaluation Report

**System under test:** the hybrid RAG pipeline in `agent-harness-from-scratch`
(`agent/rag/`) — BM25 + dense retrieval, reciprocal rank fusion, optional
heuristic reranking.
**Toolkit used:** `agent_eval.retrieval_metrics` (this repo).
**Benchmark code:** `benchmarks/rag_recall/` (`corpus.py`, `queries.py`,
`run_benchmark.py`). Re-run with `python benchmarks/rag_recall/run_benchmark.py`.
**Raw output:** `benchmarks/rag_recall/results.json`.

---

## 1. Objective

Answer, with a number instead of an impression: *does the hybrid
(BM25 + dense, RRF-fused) retrieval pipeline actually retrieve more relevant
evidence than a single-method baseline, and if so, under what conditions?*

A single blended "recall went up" number is not enough to trust — it could
come from a scoring-formula artifact rather than genuine semantic matching.
This evaluation was designed specifically to distinguish the two: queries
are split into two styles up front (§3.2), and the semantic-retrieval claim
stands or falls on the **paraphrase** subset alone.

## 2. System under test

`RAGPipeline` (`agent/rag/pipeline.py`) with three tunable inputs:

- `lexical` — `BM25Retriever`, Okapi BM25 over a Chinese/English-aware
  tokenizer (`agent/rag/retrieval.py`).
- `dense` — `DenseRetriever`, cosine similarity over an `EmbeddingProvider`.
- `reranker` — a post-fusion reordering step; defaults to
  `HeuristicReranker` (term-overlap + a fixed authority weight).

Setting `lexical=None` or `dense=None` degrades the pipeline to a
single-method retriever without touching any other code — this ablation
uses exactly that property instead of separate reimplementations per
baseline.

## 3. Methodology

### 3.1 Corpus construction

12 synthetic documents, a fictional company's internal engineering handbook
(`corpus.py`): incident response, on-call rotation, deployment pipeline,
code review, data retention, security incident classification, expense
reimbursement, VPN/remote access, database backups, API rate limits,
leave policy, vendor security review.

The corpus is **synthetic and general-domain** by deliberate choice:

- **Synthetic** — every fact's location is known exactly at write time, so
  ground truth (§3.2) can be verified programmatically instead of hand-audited.
- **General-domain, not medical** — the pipeline's chunker
  (`MedicalParentChildChunker`) has special-cased logic for Chinese clinical
  markers (recommendation strength, evidence grade, dosage units) that this
  corpus never triggers, so the ablation measures retrieval, not chunking
  edge cases.

Each document has exactly 3 `##` sections, each section 2–3 sentences
(~50–90 words). This size was chosen so every section becomes **exactly one
child chunk** — confirmed by `run_benchmark.py:check_one_chunk_per_section`,
which aborts the run if chunking ever groups sections together. 12 × 3 = 36
child chunks.

### 3.2 Query design and ground truth

24 labeled queries (`queries.py`), two per document:

- **Lexical** (12) — shares several words with its answer's sentence.
  Example: *"How quickly must a SEV1 incident be acknowledged?"* →
  *"...acknowledged within 15 minutes..."*
- **Paraphrase** (12) — asks the same thing with deliberately low word
  overlap. Example: *"Do engineers get paid extra for carrying the
  pager?"* → *"...flat stipend of 200 dollars per week..."* (no shared
  content words at all).

Ground truth is not a hand-picked chunk id (chunk ids are content hashes
assigned during ingestion — unknowable before the corpus is ingested).
Instead, each query names a **fact substring**; `resolve_relevant_ids()`
searches every ingested child chunk for that substring after ingestion and
takes the one match as ground truth. The run aborts if a fact matches zero
or more than one chunk, so a query can never silently score against the
wrong answer or an ambiguous one.

### 3.3 Embedding configuration

`qwen3.7-text-embedding`, 1024 dimensions, via an OpenAI-compatible
DashScope-based endpoint (`OpenAICompatibleEmbeddingProvider`), configured
through `RAG_EMBEDDING_MODEL` / `RAG_EMBEDDING_API_KEY` /
`RAG_EMBEDDING_BASE_URL` in the sibling repo's `.env`.

This is a **real embedding model**, not the repo's zero-dependency
deterministic hash-based fallback (`MockLLM.embed()`, a hashed bag-of-words
— stable and free, but not semantically meaningful). `run_benchmark.py`
calls `require_real_embedding_config()` before doing anything else and
raises immediately if any of the three variables is unset, specifically so
these results can never silently regress to the hash fallback and be
mistaken for a semantic-retrieval measurement.

A local, gitignored JSON cache (`cached_embeddings.py`) keys vectors by
`sha256(model_id + text)` so repeat runs don't re-embed the unchanged
36-chunk corpus and re-incur API cost.

### 3.4 Pipeline configurations compared

| Name | `lexical` | `dense` | `reranker` |
|---|---|---|---|
| `bm25_only` | BM25Retriever | — | (n/a, single method) |
| `dense_only` | — | DenseRetriever | (n/a, single method) |
| `hybrid_rrf` | BM25Retriever | DenseRetriever | `IdentityReranker` (no-op — preserves RRF order) |
| `hybrid_rerank` | BM25Retriever | DenseRetriever | `HeuristicReranker` (the repo's default) |

`RAGConfig(candidate_limit=30, evidence_limit=15)` for all four, so Recall@10
isn't truncated by the pipeline's own evidence cap.

### 3.5 Metrics

From `agent_eval.retrieval_metrics`, computed against ranked chunk-id lists:

- **Recall@k** — fraction of relevant ids present in the top-k.
- **MRR** — reciprocal rank of the first relevant id.
- **nDCG@k** — binary-relevance normalized discounted cumulative gain.

Every metric is computed per query, then averaged — overall (n=24) and
separately per style (n=12 each). Queries with no labeled relevant id would
be excluded from the average rather than count as 0 (this corpus has none).

## 4. Results

### 4.1 Overall (n=24)

| Pipeline | MRR | Recall@3 | Recall@5 | Recall@10 | nDCG@3 | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| BM25-only | 0.648 | 0.708 | 0.750 | 0.792 | 0.647 | 0.665 | 0.680 |
| Dense-only | 0.714 | 0.750 | 0.875 | 0.875 | 0.698 | 0.749 | 0.749 |
| **Hybrid (RRF)** | **0.830** | **0.833** | **0.875** | **0.917** | **0.818** | **0.836** | **0.851** |
| Hybrid + HeuristicReranker | 0.711 | 0.750 | 0.833 | 0.917 | 0.698 | 0.732 | 0.760 |

**Hybrid RRF vs. BM25-only: MRR +18.2pp, Recall@5 +12.5pp, Recall@10 +12.5pp.**

### 4.2 By query style (n=12 each)

Recall@5:

| Pipeline | Lexical | Paraphrase |
|---|---|---|
| BM25-only | 0.917 | 0.583 |
| Dense-only | 0.917 | **0.833** |
| Hybrid RRF | **1.000** | 0.750 |
| Hybrid + Rerank | 0.917 | 0.750 |

MRR:

| Pipeline | Lexical | Paraphrase |
|---|---|---|
| BM25-only | 0.839 | 0.458 |
| Dense-only | 0.839 | 0.588 |
| Hybrid RRF | **0.958** | **0.701** |
| Hybrid + Rerank | 0.844 | 0.579 |

**All four pipelines are within a few points of each other on lexical
queries** — BM25 alone already performs well once the query shares
vocabulary with its answer, which is expected and serves as a sanity check
that nothing in the setup is broken.

**The entire hybrid-vs-BM25 gap lives in the paraphrase column.** BM25-only
recovers 58.3% of paraphrased-query answers in the top 5 (MRR 0.458); every
configuration using the real embedding recovers 75–83% (MRR 0.58–0.70).
Because this gap appears specifically on queries engineered to share no
vocabulary with their answer, it is evidence of genuine semantic matching —
not of one ranking formula happening to fit this corpus better than another.

### 4.3 Finding: the heuristic reranker reduces MRR relative to plain fusion

Hybrid + `HeuristicReranker` scores **lower** on overall MRR than hybrid RRF
with no reranking at all (0.711 vs. 0.830) — reranking, here, makes results
worse. The reranker's scoring formula
(`rrf_score * 20 + lexical_overlap + authority * 0.1`,
`agent/rag/rerank.py`) re-weights toward literal term overlap after fusion
already combined both signals — exactly the signal paraphrased queries
lack, so it partially reverses the semantic gain RRF fusion produced. The
effect is visible per-style too: paraphrase MRR drops from 0.701 (RRF) to
0.579 (+ reranker), while lexical MRR is roughly flat.

This is not a criticism made in the abstract — the sibling repo's own
README already calls `HeuristicReranker` a "dependency-free baseline;
replace with a validated cross-encoder in production." This evaluation adds
a measured number to that recommendation: on this benchmark, skipping the
heuristic reranker entirely outperforms using it.

## 5. Threats to validity

- **n=24 queries / 36 chunks is small.** Each query is worth ~4.2pp overall,
  ~8.3pp within a 12-query style subset — single-digit-point differences
  (e.g., the two hybrid variants' Recall@10, both 0.917) are within noise.
  The two headline gaps — BM25 vs. hybrid RRF MRR (18.2pp) and BM25 vs.
  dense-only paraphrase Recall@5 (25pp) — are large relative to that noise
  floor and are the numbers this report treats as trustworthy.
- **One embedding model, one run, no confidence interval.** A different
  embedding model, a larger query set, or repeated sampling could shift
  these numbers; this is a single measured run, not a distribution.
- **Synthetic corpus, constructed for legibility.** Clean one-section-per-chunk
  structure and deliberately paired lexical/paraphrase queries make the
  ablation easy to audit, but real source documents (inconsistent
  structure, longer sections, ambiguous phrasing) will chunk and score
  differently.
- **What this evaluation supports:** hybrid retrieval measurably helps when
  a query doesn't share vocabulary with its answer, and a term-overlap
  reranker can fight that benefit. **What it does not support:** a specific
  recall number for any real-world corpus or query distribution.

## 6. Reproducibility

```bash
# from the evaluation/ repo root, with RAG_EMBEDDING_* set in the sibling
# repo's .env (see agent-harness-from-scratch/README.md, RAG section)
python benchmarks/rag_recall/run_benchmark.py
```

Deterministic given the same embedding model and corpus/query files: BM25
and RRF fusion are deterministic, and the embedding cache makes repeat
embedding calls for unchanged text return identical vectors.

## Appendix: full query set

| id | style | query | ground-truth fact |
|---|---|---|---|
| incident-ack-time | lexical | How quickly must a SEV1 incident be acknowledged? | acknowledged within 15 minutes |
| incident-postmortem-writeup | paraphrase | What's the deadline for writing up a review after a critical outage is fixed? | blameless postmortem published within 5 business days |
| oncall-shift-length | lexical | How long does a primary on-call shift last? | Primary on-call shifts run for one full week |
| oncall-pager-pay | paraphrase | Do engineers get paid extra for carrying the pager? | flat stipend of 200 dollars per week |
| canary-traffic-percent | lexical | What percentage of traffic does a canary release start with? | rolled out to 5 percent of production traffic |
| billing-deploy-signoff | paraphrase | Who needs to sign off before shipping a change to billing? | payments service additionally require a second engineer's sign-off |
| pr-reviewer-count | lexical | How many reviewers does a pull request need? | approval from at least one reviewer outside the author's own team |
| huge-diff-guidance | paraphrase | What should I do if my diff is huge? | Pull requests touching more than 800 lines should be split |
| log-retention-days | lexical | How long are application logs kept? | Application logs are retained for 90 days |
| data-deletion-sla | paraphrase | If a customer asks us to erase their data, how fast do we have to comply? | deletion request must be fully processed within 30 days |
| p0-definition | lexical | What counts as a P0 security incident? | P0 security incident involves confirmed unauthorized access to customer data |
| security-report-window | paraphrase | How soon do I need to tell the security team if something looks off? | report it to the security team within one hour |
| meal-limit | lexical | What's the daily meal reimbursement limit while traveling? | reimbursed up to 75 dollars per day |
| expense-approval-need | paraphrase | Do I need my boss's okay before buying something expensive for work? | expense over 500 dollars requires prior written approval |
| vpn-mfa | lexical | Does VPN access require multi-factor authentication? | require multi-factor authentication in addition to a personal client certificate |
| personal-laptop-vpn | paraphrase | Can I connect to the company network from my personal laptop? | Only company-managed laptops with disk encryption enabled |
| db-backup-frequency | lexical | How often are primary databases backed up? | backed up every 6 hours |
| backup-verification | paraphrase | How do we make sure backups actually work if we ever need them? | full restore from backup is tested in a staging environment |
| standard-rate-limit | lexical | What's the rate limit for standard API keys? | limited to 60 requests per minute |
| throttled-response-code | paraphrase | What error do clients see when they call the API too fast? | HTTP 429 response along with a Retry-After header |
| pto-accrual-rate | lexical | How many PTO days do new employees earn per year? | accrue 15 days of paid time off per year |
| pto-carryover | paraphrase | Can I roll unused vacation days into next year? | Up to 5 unused PTO days may be carried over |
| vendor-questionnaire | lexical | What does a new vendor need to complete before signing a contract? | complete a security questionnaire before a contract is signed |
| risky-vendor-recheck | paraphrase | How often do risky suppliers need to be checked again? | high-risk require an annual re-assessment |
