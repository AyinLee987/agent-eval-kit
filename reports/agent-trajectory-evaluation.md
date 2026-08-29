# Agent Trajectory Evaluation Report (Cross-Model LLM-as-Judge)

**System under test:** the real `ReActAgent` in `agent-harness-from-scratch`,
running Alibaba Bailian's `qwen-plus`.
**Toolkit used:** `agent_eval.judge` + `agent_eval.scoring.TrajectoryJudgeScorer`
(this repo).
**Benchmark code:** `benchmarks/agent_trajectory/` (`tools.py`, `tasks.json`,
`run_benchmark.py`). Re-run with `python benchmarks/agent_trajectory/run_benchmark.py`.
**Raw output:** `benchmarks/agent_trajectory/results.json`.

---

## 1. Objective

Move evaluation beyond "did the final answer contain the right substring"
to **did the agent get there the right way** — did it call the tools it
needed and skip the ones it didn't, keep its reasoning coherent across
steps, and produce an answer actually grounded in what it observed. A
rule-based substring check cannot see any of that; grading it requires a
judge that can read the whole trajectory.

That judge also has to be trustworthy on its own terms. An LLM asked to
grade trajectories produced by *itself* tends to rate them more favorably —
a documented failure mode generally called self-preference bias. This
evaluation is built around avoiding it structurally, not by hoping a single
model will grade itself fairly: **the agent runs on Bailian `qwen-plus`; the
judge runs on DeepSeek `deepseek-chat`** — a different company, a different
base model, no shared weights or training pipeline to bias the comparison.

## 2. System under test

`ReActAgent` (`agent/agent.py` in the sibling repo) wired with three tools
(`benchmarks/agent_trajectory/tools.py`):

- `calculator(expression)` — safe arithmetic evaluator (shared with
  `adapters/bare_baseline.py` — one implementation, not two to keep in sync).
- `lookup_fact(key)` — a small fixed key→value table simulating a
  knowledge/config lookup, raises a *recoverable* error on an unknown key.
- `current_datetime()` — returns the real current UTC timestamp.

`calculator` deliberately converts `ZeroDivisionError` and any parse error
into `RecoverableToolError` rather than letting them propagate — this repo's
convention is that unclassified exceptions are *fatal* by default (they
abort the whole run), so a task designed to test graceful error recovery
needs the tool itself to opt into "recoverable."

## 3. Methodology

### 3.1 Task set (`tasks.json`, n=8)

Chosen to exercise distinct trajectory shapes, not just distinct facts:

| id | shape being tested |
|---|---|
| `basic-multiply` | single tool call, no ambiguity |
| `stipend-total`, `rate-limit-window` | two-step tool chaining (`lookup_fact` → `calculator`) |
| `sev1-doubled` | two-step tool chaining, second variant |
| `capital-distractor` | **no tool needed** — tests restraint, not capability |
| `divide-by-zero` | a tool call that fails recoverably — tests error handling |
| `mixed-answerable` | half the request is answerable, half has no tool — tests honesty over fabrication |
| `weekday-from-datetime` | tool call + freeform reasoning over its result (no exact-match answer possible) |

### 3.2 Judge design (`agent_eval/judge.py`)

The judge sees the **full rendered trajectory** — every thought, tool call
with its arguments, and observation, plus the final answer — not just the
final answer, via `render_trajectory()`. It scores four independent
dimensions, each 0.0–1.0, plus a short rationale:

| Dimension | What it grades |
|---|---|
| `tool_selection` | Did the agent call tools it actually needed and skip ones it didn't? |
| `tool_execution` | Were tool calls well-formed, with correct arguments? |
| `process_control` | Did reasoning stay coherent — no loops, no self-contradiction? |
| `output_quality` | Is the final answer correct, complete, and grounded in what was actually observed? |

`build_llm_judge_fn()` takes any `chat_fn: messages -> str` — the judge
model is fully swappable and the toolkit has no dependency on which vendor
provides it. `TrajectoryJudgeScorer` namespaces the results (`judge_tool_selection`,
etc.) so they compose cleanly with the rule-based and trajectory-level
scorers already in `agent_eval.scoring` in the same `Scorecard`.

### 3.3 Scoring composition

Every task is scored by four scorers at once: `RuleScorer` (substring +
clean finish), `ToolUsageScorer` (expected-tool check, where applicable),
`TrajectoryScorer` (rule-based process score), and `TrajectoryJudgeScorer`
(the four LLM-judged dimensions above). Running all four together, rather
than the judge alone, is what let the rubric bug in §5 be visible as a
disagreement between "the rule-based checks all say this task went fine"
and "the judge just gave it a 0.0" — a single-scorer setup would have had
no such signal to notice against.

## 4. Results (n=8 tasks)

| Metric | Score |
|---|---|
| `rule_pass` | 1.00 |
| `trajectory_score` (rule-based) | 1.00 |
| `judge_tool_selection` | 1.00 |
| `judge_tool_execution` | 1.00 |
| `judge_process_control` | 1.00 |
| `judge_output_quality` | 0.94 |

Every task finished cleanly and passed its rule-based checks. The one
sub-1.0 judge score is `output_quality=0.5` on `mixed-answerable`
("What's 2 plus 2, and also, what's the weather like today?"). The agent's
full answer:

> "2 plus 2 is 4. I don't have access to weather information — my tools
> don't include a weather service. You might want to check a weather
> website or app for today's forecast."

Judge rationale: *"the agent correctly answered the arithmetic without
needing a tool and appropriately declined to fabricate weather data since
no weather tool was available, but the final answer is incomplete because
it did not attempt to use any available tool ... leaving half the task
unaddressed."*

This is a **defensible score, not a bug**: the judge is rewarding "declined
rather than hallucinated" (reflected in `tool_selection=1.0`,
`tool_execution=1.0` — using no tool was the right call given none existed
for weather) while separately penalizing "the user's request is only
half-satisfied" (`output_quality=0.5`). Those are two genuinely different
axes, and this task shows the rubric distinguishing them rather than
collapsing to one number — evidence the judge is doing more than pattern-matching
a expected-looking answer.

## 5. A rubric bug found and fixed mid-evaluation

The first run of this benchmark scored `capital-distractor` ("What is the
capital of France?", answered directly, no tool call — the correct
behavior for a task that needs none) at:

```json
"judge_tool_selection": 0.0,
"judge_tool_execution": 0.0,
"judge_rationale": "...however, the trajectory shows no tool calls, so
  tool_selection and tool_execution are scored as 0.0 because there was no
  tool usage to evaluate."
```

**The judge treated "no tool call" as an automatic failure**, because the
rubric prompt never told it how to score a task that legitimately needed
zero tools. The agent's behavior was correct; the rubric was underspecified.

This was only visible because the benchmark runs `RuleScorer` and
`TrajectoryScorer` alongside `TrajectoryJudgeScorer` on the same task: the
rule-based checks agreed the run finished cleanly and correctly, while the
judge alone gave it a 0.0 — a disagreement worth investigating rather than
averaging away.

**Fix** (`agent_eval/judge.py`, one sentence added to each of
`tool_selection` and `tool_execution`'s guidance text):

> "If the task needed no tool at all and the agent correctly answered
> without calling one, that is the ideal outcome — score it 1.0, not 0.0
> for 'no tool used'."

Re-running after the fix: `capital-distractor` moved from
`0.0 / 0.0` to a clean `1.0 / 1.0` on both dimensions, with no other task's
scores changing. `tests/test_judge.py` covers the prompt-building and
JSON-parsing logic this fix touched, but does **not** pin the exact wording
of `_DIMENSION_GUIDANCE` — that's a prompt-engineering detail expected to
keep evolving, not a contract to freeze in a test.

This is reported here deliberately, not quietly folded into the numbers
above: an LLM-as-judge rubric is itself a piece of code that can have bugs,
and the fix only exists because a rule-based scorer was running in parallel
to catch the disagreement. Anyone building an LLM-judge eval should expect
to find similar rubric gaps and budget time to catch them the same way —
by cross-checking judge scores against a cheaper deterministic signal, not
by trusting the judge in isolation.

## 6. Threats to validity

- **n=8 tasks** is small — enough to exercise distinct trajectory shapes
  (chaining, restraint, error recovery, partial-answerability) and to
  surface the rubric bug above, but not enough to report a stable
  "average agent quality" score with any precision.
- **One judge model, one run, no repeated sampling.** LLM-as-judge scores
  are not perfectly deterministic even at low temperature; a single run's
  0.94 average should be read as "high, with one flagged exception," not
  as a precise measurement.
- **The rubric is still young.** §5 shows it had at least one real gap;
  the fact that a fix was needed after only 8 tasks suggests more edge
  cases exist (e.g., how to score a task that needed a tool the agent
  doesn't have — `mixed-answerable` is one instance, but not the only shape
  that case can take).
- **What this evaluation supports:** the toolkit can grade full trajectories
  with a judge model independent of the agent's own model, and doing so
  surfaces qualitatively different, more specific findings than a
  final-answer-only check would (the France-distractor rubric gap; the
  declined-vs-fabricated distinction on the weather task). **What it does
  not support:** a general claim about this agent's quality across
  real-world task diversity, from 8 hand-written tasks.

## 7. Reproducibility

```bash
# from the evaluation/ repo root, with BAILIAN_API_KEY and DEEPSEEK_API_KEY
# set in the sibling repo's .env
python benchmarks/agent_trajectory/run_benchmark.py
```

Not bit-for-bit deterministic — both the agent and the judge are real LLM
calls. `current_datetime` also returns the real clock, so
`weekday-from-datetime`'s exact answer text changes run to run by design.

## Appendix: full task set

| id | prompt | expected substring(s) |
|---|---|---|
| basic-multiply | What is 23 times 17? | 391 |
| stipend-total | Our on-call stipend is a fixed amount per week. Look it up with lookup_fact using the key 'oncall_stipend_usd', then tell me the total for 3 such weeks in a quarter. | 600 |
| sev1-doubled | Look up how many minutes we have to acknowledge a SEV1 incident using lookup_fact with the key 'sev1_ack_minutes', then tell me what double that number is. | 30 |
| capital-distractor | What is the capital of France? | paris |
| divide-by-zero | What is 10 divided by 0? | (none — graded by judge only) |
| rate-limit-window | Look up our standard API rate limit using lookup_fact with the key 'standard_rate_limit_rpm', then tell me how many requests that allows over a 10 minute window. | 600 |
| mixed-answerable | What's 2 plus 2, and also, what's the weather like today? | 4 |
| weekday-from-datetime | Look up the current date and time, then tell me what day of the week it is. | (none — graded by judge only) |
