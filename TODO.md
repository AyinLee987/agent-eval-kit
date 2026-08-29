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
- [ ] **Multi-agent latency benchmark.** Drive
      `MultiAgentOrchestrator.spawn_subagent` + `wait_subagents` against N
      independent worker tasks and compare wall-clock time to a sequential
      loop of `agent.run()` calls, across a few `max_parallel_tasks` values.
      Also measure failure isolation: inject a fatal error into one worker
      and confirm the others still complete (the concurrency benchmark's
      `failure_count` already reports this shape).
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
- [ ] **② Robustness (paraphrase/translation → answer similarity).** For a
      set of base queries, generate 3–4 variants each (paraphrase +
      translation to 1–2 other languages), run the agent on every variant,
      embed the answers (reuse `OpenAICompatibleEmbeddingProvider` +
      `CachedEmbeddingProvider` from the RAG benchmark — no new embedding
      infra needed), and average pairwise cosine similarity per base query.
      Higher = more robust. Decide up front: variants pre-authored and
      cached (reproducible) vs. generated fresh each run (more realistic,
      less deterministic) — leaning toward pre-authored, matching the
      RAG benchmark's lexical/paraphrase query design.
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
- [ ] **⑤ Business-scenario evaluation.** Not yet scoped. `Conversation`/
      `ConversationHarness` now exists (built for ①) — likely a composite
      of multi-turn scenarios and specific tool combinations on top of it.

## Toolkit hardening (before calling this a "full version")

- [ ] Semantic-similarity scorer for open-ended answers that doesn't require
      an LLM judge call (e.g. embedding cosine similarity against a reference
      answer) — cheaper and more deterministic than `LLMJudgeScorer` for CI.
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
