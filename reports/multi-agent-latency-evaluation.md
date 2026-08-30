# Multi-Agent Latency & Failure-Isolation Evaluation Report

**System under test:** the real `MultiAgentOrchestrator` in
`agent-harness-from-scratch`, dispatching real `ReActAgent` Workers running
Alibaba Bailian's `qwen-plus`.
**Toolkit used:** no new `agent_eval` module — this benchmark drives the
sibling repo's orchestrator directly and reports timing it already exposes
(`SubagentTask.created_at/started_at/finished_at`,
`SubagentResult.success/error_type`) rather than routing through
`EvalHarness`, since the thing being measured (wall-clock concurrency, not
per-task correctness scoring) doesn't fit that harness's shape.
**Benchmark code:** `benchmarks/multi_agent_latency/` (`tasks.json`,
`tools.py`, `run_benchmark.py`). Re-run with
`python benchmarks/multi_agent_latency/run_benchmark.py`.
**Raw output:** `benchmarks/multi_agent_latency/results.json`.

---

## 1. Objective

`MultiAgentOrchestrator` exists specifically to run independent Worker
tasks concurrently. Two things about that claim had never been measured
with real data:

1. **How much wall-clock time does concurrent dispatch actually save**,
   using real network calls rather than a synthetic `time.sleep` stand-in
   (a synthetic delay can't demonstrate the actual mechanism — parallelism
   helps here because it overlaps *real* API round-trip wait time, and only
   real calls can show that honestly)?
2. **Does the orchestrator's specific failure-isolation claim hold** — the
   README states "a child fatal error terminates only that child," but
   this had not been verified under an actual concurrent run with real
   data.

## 2. A load-bearing implementation fact

`MultiAgentOrchestrator.__init__` builds its own `ThreadPoolExecutor`,
sized by `RunBudget.max_parallel_tasks`, at construction time
(`agent/multi_agent/orchestrator.py`). This is what "how parallel" actually
means for this system — it is not something this benchmark wraps around
the orchestrator from outside; a fresh orchestrator instance (and thus a
fresh, differently-sized thread pool) is required to test a different
`max_parallel_tasks` value.

This also decided how the benchmark measures orchestrated dispatch:
`spawn_subagent` is called directly, six times, followed by one
`wait_subagents` call — not through a real Leader LLM deciding whether and
how to delegate. That isolates pure dispatch/wait infrastructure latency
from an entirely separate variable (whether and how a model chooses to
delegate), which is not what this benchmark is about.

## 3. Methodology

### 3.1 Tasks (`tasks.json`, n=6)

Six independent, single-tool-call tasks (`calculator` ×3, `lookup_fact`
×2, `current_datetime` ×1) — short and roughly uniform in difficulty, so
the wall-clock comparison isn't skewed by one unusually hard task, and
dispatched identically in every arm below.

### 3.2 Latency comparison

- **Sequential baseline**: a plain Python loop — build a fresh
  `ReActAgent` (fresh `BailianLLM`, fresh `ToolRegistry`) and call `.run()`
  for each of the 6 tasks, one after another. No orchestrator involved at
  all.
- **Orchestrated**: a fresh `MultiAgentOrchestrator` per
  `max_parallel_tasks` value tested (1, 2, 3, 6) — inside one
  `leader_scope()`, `spawn_subagent("worker", ...)` is called six times
  (each returns immediately with a task id; the actual Worker run happens
  on the orchestrator's own thread pool), then one `wait_subagents(...)`
  call blocks until all six finish. Wall-clock is measured around exactly
  that spawn-all-then-wait sequence.

### 3.3 Failure isolation

Two distinct failure shapes, because they exercise different code paths:

- **Pre-flight**: `spawn_subagent` with an unregistered role name. This is
  rejected synchronously, before a task record is even created — the
  question is whether one bad request derails the rest of the batch.
- **Mid-run**: a deliberately-broken Worker role (`broken-worker`, whose
  only tool, `tools.py`'s `broken_tool`, always raises `FatalToolError`) is
  spawned *alongside* 5 normal Workers in the same batch, all running
  concurrently. This is the shape the README's claim is actually about —
  a failure while other tasks are genuinely in flight, not a failure that
  never got scheduled.

## 4. Results

### 4.1 Latency (n=6 tasks)

| Mode | Wall-clock | Speedup vs. sequential | Success |
|---|---|---|---|
| Sequential (baseline) | 19.73s | 1.00x | 6/6 |
| Orchestrated, `k=1` | 15.99s | 1.23x | 6/6 |
| Orchestrated, `k=2` | 8.49s | 2.33x | 6/6 |
| Orchestrated, `k=3` | 4.93s | 4.00x | 6/6 |
| Orchestrated, `k=6` | 3.03s | 6.50x | 6/6 |

Per-task queue/exec timing (from `SubagentTask.created_at/started_at/
finished_at`, `results.json`) confirms the mechanism directly rather than
just the aggregate number:

| `k` | Per-task queue time pattern |
|---|---|
| 1 | Fully serialized: task *n*'s queue time ≈ sum of all prior tasks' exec times. |
| 2 | Tasks 1–2 start immediately; 3–4 wait ~task-1-and-2's exec time; 5–6 wait again. |
| 3 | Tasks 1–3 start immediately (`queue≈0`); 4–6 wait ~2.0–2.2s — almost exactly the first batch's exec duration. |
| 6 | All 6 start immediately (`queue=0` for every task); total wall-clock ≈ the single slowest task (3.03s). |

This is exactly what a thread pool with `k` slots draining a queue of 6
items should look like — the speedup isn't a black box, it's visibly a
smaller number of sequential "waves."

### 4.2 Why `k=1` beat the sequential baseline (1.23x) — and why that's not the finding

Both `k=1` and the sequential baseline run one task at a time with no
actual concurrency — they should, in principle, take about the same time.
They didn't: 15.99s vs. 19.73s. Looking at individual call durations
explains the gap without needing to invoke anything structural: the
baseline's 6 calls averaged 3.29s each, while `k=1`'s averaged 2.67s each
— a run-to-run difference in real API latency, not a difference in what
either arm was doing (both build a fresh `BailianLLM`/`ReActAgent` per
task; neither reuses any state).

With only 6 real network calls per arm and a single run, that's within the
range of ordinary call-to-call latency variance for a real API — not
strong enough evidence to claim the orchestrator path is inherently faster
than a bare loop even absent parallelism. Confirming or ruling that out
would need repeated trials averaging out per-call noise, which this run
doesn't attempt. The headline finding is the `k=2` through `k=6` curve,
where the speedup tracks `max_parallel_tasks` cleanly and the per-task
timing data explains exactly why.

### 4.3 Failure isolation

- **Pre-flight (unknown role)**: rejected synchronously
  (`RecoverableToolError`, caught by the calling code before any task
  existed). The other 6 real tasks all completed successfully afterward
  (6/6) — confirmed a malformed delegation request doesn't block or corrupt
  the rest of a batch.
- **Mid-run (fatal tool error)**: the broken Worker failed cleanly
  (`success=False`, `stop_reason=fatal_tool_error`) while running
  *concurrently* with 5 normal Workers on the same orchestrator instance —
  all 5 of which succeeded (5/5). `wait_subagents` returned normally for
  the whole batch; nothing hung, and the fatal failure in one thread didn't
  propagate to or block the others.

Both results support the README's specific claim, now under an actual
concurrent run rather than the single-task case the claim is usually
illustrated with.

## 5. Threats to validity

- **n=6 tasks, one run per condition, no repeated trials.** §4.2 is the
  direct consequence: a single small sample can't distinguish "the
  orchestrator has some inherent per-call efficiency" from "this
  particular run's API calls happened to be faster." The `k=2..6` speedup
  curve is far larger than that noise band and is corroborated by the
  per-task queue-time mechanism in §4.1, which is why it's reported as the
  finding and §4.2 isn't.
- **Uniform, short, single-tool tasks.** All 6 tasks take a similar amount
  of time and involve one tool call each. A more heterogeneous task mix
  (some tasks far slower than others) would show a different, likely
  smaller, speedup at a given `k`, since the slowest task increasingly
  dominates wall-clock time as parallelism increases (visible already at
  `k=6`, where total time equals the single slowest task's duration, not
  zero).
- **Two failure shapes, not an exhaustive fault-injection suite.** Only one
  concrete instance of each (unregistered role; a tool that always raises
  `FatalToolError`) was tested — not, for example, a Worker that times out
  (`RunBudget.subagent_timeout_seconds`), or multiple simultaneous
  failures. Both tested shapes passed cleanly; that supports the README's
  claim for the cases tested, not a blanket guarantee against every failure
  mode the orchestrator could encounter.
- **What this evaluation supports:** real concurrent Worker dispatch
  delivers real, roughly-proportional-to-`max_parallel_tasks` wall-clock
  savings for independent, similarly-sized tasks, and a fatal failure in
  one concurrently-running Worker does not block or corrupt the others.
  **What it does not support:** a claim about speedup under a realistic
  heterogeneous task mix, or failure isolation under failure modes beyond
  the two tested here.

## 6. Reproducibility

```bash
# from the evaluation/ repo root, with BAILIAN_API_KEY set in the sibling
# repo's .env
python benchmarks/multi_agent_latency/run_benchmark.py
```

Not bit-for-bit deterministic — every task in every arm is a real LLM call;
wall-clock numbers will vary run to run with real network conditions (see
§4.2).
