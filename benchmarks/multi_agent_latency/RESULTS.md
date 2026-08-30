# Multi-agent latency + failure isolation — results

Run with `python benchmarks/multi_agent_latency/run_benchmark.py`. Full
per-task timing and failure-isolation detail are in `results.json`; the
full write-up (including why `k=1` beat the sequential baseline, which
isn't the headline finding) is
`../../reports/multi-agent-latency-evaluation.md`.

## Setup

- **Agent**: real `ReActAgent` (`agent-harness-from-scratch`), LLM = Bailian
  `qwen-plus`, same 3 tools as the other benchmarks
  (`calculator`, `lookup_fact`, `current_datetime`).
- **6 independent single-tool tasks** (`tasks.json`), dispatched two ways:
  - **sequential baseline** — a plain loop of `ReActAgent.run()` calls, no
    orchestrator, no threading.
  - **orchestrated** — the real `MultiAgentOrchestrator` (its own internal
    `ThreadPoolExecutor`, sized by `RunBudget.max_parallel_tasks`) via
    direct `spawn_subagent` + one `wait_subagents` call — bypassing a real
    Leader LLM's own delegate-or-not decision, to measure pure dispatch
    infrastructure latency rather than model decision variance.
- **Real API calls throughout** — no synthetic `time.sleep` stand-ins, since
  the entire point is that parallelism benefit comes from overlapping real
  network wait time.
- **Failure isolation**: two tests, verifying the sibling repo's README
  claim ("a child fatal error terminates only that child") with real data
  instead of taking it on faith — a pre-flight validation failure (unknown
  Worker role, rejected before a task is even queued) and a mid-run fatal
  tool error inside an already-running Worker (a deliberately-broken tool,
  `broken_tool`, that always raises `FatalToolError`).

## Results

| Mode | Wall-clock | Speedup | Success |
|---|---|---|---|
| Sequential (baseline) | 19.73s | 1.00x | 6/6 |
| Orchestrated, `max_parallel_tasks=1` | 15.99s | 1.23x | 6/6 |
| Orchestrated, `max_parallel_tasks=2` | 8.49s | 2.33x | 6/6 |
| Orchestrated, `max_parallel_tasks=3` | 4.93s | 4.00x | 6/6 |
| Orchestrated, `max_parallel_tasks=6` | 3.03s | 6.50x | 6/6 |

Speedup scales cleanly with `max_parallel_tasks` — at `k=6` (full
parallelism for 6 tasks), wall-clock time is bounded by the single slowest
task (3.03s) instead of the sum of all six (19.73s). Per-task queue-vs-exec
timing (`results.json`) confirms the mechanism directly: at `k=3`, the
first 3 tasks start immediately (`queue_seconds≈0`) while the last 3 wait
almost exactly as long as the first batch took to finish — precisely what
a 3-slot thread pool draining a 6-item queue should look like.

`k=1` beating the sequential baseline (1.23x, despite both being
effectively one-call-at-a-time) is a real observation in this run's data,
not a rounding artifact — but it isn't the benchmark's finding either; see
the full report for why n=6-per-arm real API latency variance is the more
likely explanation than "the orchestrator is inherently faster than a
bare loop even with no parallelism."

## Failure isolation — both hold up

- **Pre-flight (unknown role)**: rejected before being queued, and the
  other 6 real tasks all completed successfully (6/6) — one bad request
  doesn't block or crash the batch.
- **Mid-run (fatal tool error)**: the deliberately-broken Worker failed
  cleanly (`success=False`, `stop_reason=fatal_tool_error`) while running
  *concurrently* with 5 normal Workers — all 5 of which still succeeded
  (5/5). The README's "a child fatal error terminates only that child"
  claim held under an actual concurrent run, not just in the single-task
  case it's usually described for.
