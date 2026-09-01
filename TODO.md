# Roadmap

Tracking the path from "light version" (current) to a "full version" worth
expanding on for a resume/portfolio writeup.

## Done (light version)

- [x] Framework-agnostic contract: `AgentOutcome` / `TrajectoryStep` / `ToolCall`
- [x] Composable scorers: rule-based, tool-usage, trajectory-level, LLM-judge (pluggable judge function)
- [x] `EvalHarness` + `Scorecard` driven by scorer output, not a fixed schema
- [x] Retrieval metrics decoupled from any retriever: Recall@K, MRR, nDCG, `evaluate_retrieval`
- [x] Concurrency benchmark: serial-vs-parallel speedup + failure isolation
- [x] Minimal CLI (`python -m agent_eval run ...`)
- [x] Self-contained baseline agent + adapter (zero deps, used by the test suite)
- [x] Lazy adapter for the sibling `agent-harness-from-scratch` ReActAgent
- [x] Unit tests for every module (21 tests, all passing)

## Next: turn the toolkit into actual resume-grade numbers

- [x] **RAG recall benchmark.** Done — see `benchmarks/rag_recall/RESULTS.md`.
      12 synthetic documents, 24 labeled queries (lexical + paraphrase),
      real `qwen3.7-text-embedding` vectors (not the repo's hash fallback),
      four pipeline configs. Headline: hybrid RRF improves MRR +18.2pp and
      Recall@5 +12.5pp over BM25-only, concentrated entirely in paraphrased
      queries; the default `HeuristicReranker` actually *reduces* MRR
      relative to plain RRF fusion because it re-weights toward lexical
      overlap. Read the caveats section before citing a number from it —
      n=24 is small, one embedding model, one run.
- [x] **Statistical confidence for retrieval metrics.** Done — new
      `agent_eval/stats.py` (`bootstrap_ci`, `paired_bootstrap_test`,
      dependency-free bootstrap resampling, no scipy/numpy). Wired into
      `benchmarks/rag_recall/run_benchmark.py`: every pipeline's MRR now
      reports a 95% CI, and the two headline comparisons (BM25 vs. hybrid
      RRF; hybrid RRF vs. +reranker) are paired-tested — both clear p<0.05
      even at n=24. Reusable by any other benchmark's per-item scores, not
      RAG-specific.
- [x] **RAG recall on a published benchmark (BEIR NFCorpus).** Done — see
      `benchmarks/rag_recall_beir/RESULTS.md`. Directly addresses
      `benchmarks/rag_recall`'s "synthetic, self-authored corpus" caveat:
      3,633 real biomedical documents, 323 real relevance-judged test
      queries (graded 1/2, from the original 2016 BEIR release), fetched
      once via `download_nfcorpus.py` (HuggingFace `datasets-server`, no
      `datasets`/pandas dependency) into checked-in JSON. Same four-pipeline
      ablation as the synthetic benchmark, same `agent_eval.stats` CI/
      significance testing, ~13x the query count. Found real-corpus
      documents split into several chunks each under the sibling repo's
      `MedicalParentChildChunker` (avg 2.1 chunks/doc, vs. the synthetic
      corpus's one-section-one-chunk-by-design) — `build_cases()` treats
      every chunk of a relevant document as relevant, the correct ground
      truth for document-level qrels split into passages. **Results (n=323):**
      hybrid RRF vs. BM25-only replicates the synthetic benchmark's headline
      finding (+11.9pp MRR, 0.448→0.567, p=0.000). Two findings that don't
      carry over as cleanly: dense-only edges out hybrid RRF on MRR here
      (0.577 vs. 0.567, though hybrid still wins Recall/nDCG — their CIs
      overlap heavily, read as "roughly tied" not "dense wins"), and the
      synthetic benchmark's significant reranker-hurts-MRR finding (p=0.007
      at n=24) is directionally the same but no longer significant at n=323
      (p=0.262) — a real, measured instance of "a small effect can need
      more than 24 queries to separate from noise," not a contradiction.
- [x] **LLM-as-reranker, on the `LLM_ReRanker` branch.** Done — see
      `benchmarks/rag_recall_beir/RESULTS_llm_rerank.md`. New
      `llm_reranker.py` wires `deepseek-chat` into the sibling repo's
      existing `CallableReranker` adapter (no harness changes needed) --
      one call per query, listwise-scores its up-to-30 fused RRF
      candidates. First (zero-shot) prompt had a **41% degenerate-response
      rate** (the model literally echoing `[json array of 30 numbers]`
      instead of doing the task) -- fixed by rewriting the prompt as
      one-shot (one full worked example instead of a one-line format
      description): **0/323 failures** after. Result: LLM rerank tops
      every pipeline on every metric, and clears paired-bootstrap
      significance (p<0.05, mostly p<0.001) against hybrid RRF,
      hybrid+HeuristicReranker, and dense-only on Recall@5/10 and
      nDCG@5/10 -- the one metric that stays non-significant is MRR, same
      "noisiest metric on this corpus" pattern as the rest of this
      benchmark. Cost: ~2.2s/query, 6-12x slower than the local pipelines
      -- a real latency tradeoff, not a free win.
- [x] **Trajectory evaluation + cross-model LLM-judge.** Done — see
      `benchmarks/agent_trajectory/RESULTS.md` and
      `reports/agent-trajectory-evaluation.md`. Real `ReActAgent` (Bailian
      `qwen-plus`) graded by an independent DeepSeek judge (different
      vendor/model, to avoid self-preference bias) across 4 trajectory
      dimensions (`tool_selection`, `tool_execution`, `process_control`,
      `output_quality`) via the new `agent_eval.judge` module +
      `TrajectoryJudgeScorer`. 8 tasks, avg 0.94–1.00. Also caught and fixed
      a real rubric bug (judge scored "correctly used no tool" as 0.0) —
      only visible because rule-based scorers ran alongside the judge on
      the same tasks.
- [x] **Multi-agent latency benchmark.** Done — see
      `benchmarks/multi_agent_latency/RESULTS.md` and
      `reports/multi-agent-latency-evaluation.md`. Real `spawn_subagent` +
      `wait_subagents` (direct calls, bypassing a Leader LLM's own
      delegate-or-not decision, to isolate pure dispatch latency) against 6
      independent single-tool tasks, real Bailian calls throughout — no
      synthetic `time.sleep` stand-ins. Speedup vs. a sequential
      `agent.run()` loop tracked `max_parallel_tasks` cleanly: 2.33x at
      k=2, 4.00x at k=3, 6.50x at k=6, all explained directly by per-task
      queue/exec timing (`SubagentTask.created_at/started_at/finished_at`).
      Also verified, with real data for the first time, the sibling repo's
      README claim "a child fatal error terminates only that child" — both
      a pre-flight validation failure (unknown role) and a mid-run
      `FatalToolError` inside one Worker running concurrently with 5 normal
      ones left the other tasks unaffected (6/6 and 5/5 respectively).
- [ ] **Context-compression ablation.** Run the same task set through
      `EvalHarness` with and without `ContextCompressor` wired into the
      agent; report avg tokens/task and success rate side by side to show
      the compression doesn't cost accuracy.
- [ ] **Prompt-injection guard precision/recall.** Build ~20-30 injection
      payloads and ~20-30 benign tool outputs; add a scorer or standalone
      script that reports `ToolOutputGuard`'s TPR/FPR on that set.

## Agent-level evaluation plan (5 categories)

The broader plan for evaluating the agent itself (not just its RAG
retrieval), as scoped in conversation. Ordered by what's done, not by
priority — ④ was deliberately built first because it had the least
architectural risk (no new core types needed) and the cross-model judge
setup (Bailian agent / DeepSeek judge) was immediately available.

- [x] **④ Full trajectory evaluation + cross-model LLM-judge.** Done, see above.
- [x] **① Conversational ability (single-turn + multi-turn).** Done — see
      `benchmarks/conversational/RESULTS.md` and
      `reports/conversational-evaluation.md`. Extended `agent_eval` with a
      first-class multi-turn concept as planned: `types.Turn` /
      `ConversationOutcome`, `conversation_harness.ConversationHarness`
      (one agent per conversation, not per turn), `conversation_scoring.
      ConversationJudgeScorer` (knowledge retention + conversation
      completeness), and a new single-turn `scoring.AnswerRelevancyScorer`.
      Discovered `ReActAgent.run()` doesn't persist state across calls by
      itself — multi-turn state is threaded through the sibling repo's
      existing `ContextProvider` hook via the new
      `adapters/conversation_history.py`, no sibling-repo changes needed.
      7 conversations (Bailian agent / DeepSeek judge): 1.00/1.00 on both
      conversation-level dimensions, answer_relevancy 0.98 after fixing a
      **second instance of the same rubric-design bug** found in ④ (a
      judge implicitly assuming every turn is a question). Role adherence
      deliberately deferred to ③ (adversarial persona-holding overlaps with
      safety testing more than with this batch's straightforward scenarios).
- [x] **② Robustness (paraphrase/register/translation → answer
      similarity).** Done — see `benchmarks/robustness/RESULTS.md` and
      `reports/robustness-evaluation.md`. Added `agent_eval/similarity.py`
      (`cosine_similarity` / `average_pairwise_similarity`, dependency-free)
      to the toolkit. 8 base questions × 4 phrasings each (base, same-
      language paraphrase, formal/politeness-noise register shift, English
      translation) = 32 single-turn tasks, real Bailian agent + real
      `qwen3.7-text-embedding`. `rule_pass`/`used_expected_tool` both a
      clean 1.00 across all 32 — rewording never changed *what* the agent
      computed. Overall avg pairwise similarity 0.916, but the lowest-
      scoring question (`chain-stipend`, 0.798) turned out **not** to be an
      inconsistency: all four phrasings computed the same correct total,
      the embedding was picking up on answer verbosity/formatting, not
      correctness — confirmed the same pattern on two other below-average
      questions too. Exactly the caveat the plan flagged going in
      ("embedding similarity is coarse for short factual answers"), now
      with three concrete worked examples instead of a hypothetical one.
- [ ] **③ Safety (public datasets).** Candidates: AdvBench, HarmBench,
      TruthfulQA, Do-Not-Answer (English); SafetyBench or Flames (Chinese —
      more relevant given the sibling project's Chinese RAG focus). Sample
      100–200 prompts rather than a full dataset. Three-part split: direct
      harmful-request refusal rate; (reusing the sibling project's
      `ToolOutputGuard`) resistance to instructions injected via tool
      output — overlaps with the "prompt-injection guard precision/recall"
      item below; and **role adherence under adversarial pressure**
      (deferred here from ①) — give the agent a persona/scope restriction
      and use `ConversationHarness` (now built) to test whether a multi-turn
      conversation can talk it out of that persona.
- [x] **⑤ Business-scenario evaluation — scoped as medical QA (risk-tiered
      triage), replacing the earlier generic "business scenario" framing.**
      Done — see `benchmarks/medical_qa/RESULTS.md` and
      `reports/medical-qa-evaluation.md`. A real ReActAgent with one tool
      (RAG-as-a-tool, `medical_evidence_search`) over a governed hybrid
      pipeline retrieving a 6-document knowledge base adapted from real
      MedlinePlus (NIH) pages (5 low-risk conditions + 1 emergency
      warning-signs document, each citation carrying its real source URL
      end-to-end). Agent follows an explicit risk-tiering policy: check
      against warning signs first and escalate, not diagnose, on a match;
      otherwise a hedged/cited tentative judgment only if the knowledge
      base actually has a matching entry; plain "no evidence for this"
      otherwise. New `clinical_accuracy` conversation-judge dimension
      (`agent_eval/judge.py`) added alongside the existing retention/
      completeness pair. 4/6 conversations scored a clean 1.0/1.0/1.0,
      including correctly *revising* an assessment mid-conversation once a
      correction turned a low-risk symptom into a red flag, and correctly
      separating a red flag from a low-risk symptom in one mixed message.
      `ungrounded-question` originally caught the agent fabricating a
      diagnosis (rheumatoid arthritis, specific lab tests) for a symptom
      with zero matching retrieved evidence, **and falsely claiming the
      fabrication came from retrieval** — verified by directly querying
      the pipeline. Adding an explicit persona to the system prompt
      ("hospital triage/pre-screening assistant, not a doctor") afterward
      fixed that specific case (0.0 → 0.90) — but the same re-run caught a
      *new* regression in a different scenario (`real-patient-phrasing-
      check`, tool usage dropped to 0): the agent skipped the mandatory
      search step entirely for a meta-question about a condition rather
      than a first-person symptom report, exposing an ambiguity in the
      prompt's own "before responding to a symptom description" wording.
      Reported as-is rather than patched again — the point being that a
      prompt fix aimed at one failure mode needs re-testing against the
      *whole* scenario set, not just the case it targeted. Also discovered
      mid-project that a real HuggingFace
      medical dataset's "encyclopedia" content is actually unreliable
      crowd-sourced forum Q&A, unsuitable as corpus content — resolved by
      using authoritative MedlinePlus pages for the knowledge base while
      still using the dataset's real patient-phrased questions (Apache-2.0)
      for one scenario's realism.

## Toolkit hardening (before calling this a "full version")

- [ ] Semantic-similarity scorer for open-ended answers that doesn't require
      an LLM judge call (e.g. embedding cosine similarity against a reference
      answer) — cheaper and more deterministic than `LLMJudgeScorer` for CI.
      The primitive now exists (`agent_eval/similarity.py`, built for ②
      robustness) — what's missing is a `Scorer`-shaped wrapper around it
      that takes a reference answer per task.
- [ ] A second real-framework adapter (e.g. a LangChain agent) to prove the
      "framework-agnostic" claim against something other than an internal
      baseline.
- [ ] CLI: judge-scorer wiring, a `retrieval` subcommand, a `concurrency`
      subcommand.
- [ ] Package it properly (`pyproject.toml`, versioned release) instead of
      running from a checkout.
- [ ] GitHub Actions CI (mirrors the sibling project's).
- [ ] Publish the benchmark results (numbers + methodology) in this README
      once the "Next" section above has real runs behind it.
