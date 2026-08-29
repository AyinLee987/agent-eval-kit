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

- [ ] **RAG recall benchmark.** Build a small labeled corpus (synthetic docs +
      ~30-50 queries with known relevant chunk ids). Run `evaluate_retrieval`
      against the sibling project's `RAGPipeline` in four configurations —
      BM25-only, dense-only, hybrid (RRF), hybrid + reranker — and report
      Recall@5/@10, MRR, nDCG@10 for each. This is the number that answers
      "how much did hybrid retrieval improve recall."
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
