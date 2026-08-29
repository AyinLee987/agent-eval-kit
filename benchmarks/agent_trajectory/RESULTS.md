# Trajectory + cross-model LLM-judge — results

Run with `python benchmarks/agent_trajectory/run_benchmark.py`. Full
per-task scores and judge rationale are in `results.json`; the full
write-up (including a rubric bug found and fixed during this run) is
`../../reports/agent-trajectory-evaluation.md`.

## Setup

- **Agent**: real `ReActAgent` (`agent-harness-from-scratch`), LLM = Bailian
  `qwen-plus`, 3 tools (`calculator`, `lookup_fact`, `current_datetime`).
- **Judge**: DeepSeek (`deepseek-chat`) — a **different vendor and base
  model** than the agent, specifically so the judge isn't grading its own
  outputs (self-preference bias).
- **8 tasks** (`tasks.json`): single tool call, two multi-step
  tool-chaining tasks, a recoverable tool error (divide by zero), a
  no-tool-needed distractor, a partially-unanswerable mixed request, and a
  tool + freeform-reasoning task.
- **Scoring**: `RuleScorer` + `ToolUsageScorer` + `TrajectoryScorer` (all
  from the RAG-benchmark toolkit) plus the new `TrajectoryJudgeScorer`,
  which grades the **entire trajectory** — not just the final answer —
  across four dimensions: `tool_selection`, `tool_execution`,
  `process_control`, `output_quality`.

## Results (n=8 tasks)

| Metric | Score |
|---|---|
| rule_pass | 1.00 |
| trajectory_score | 1.00 |
| judge_tool_selection | 1.00 |
| judge_tool_execution | 1.00 |
| judge_process_control | 1.00 |
| judge_output_quality | 0.94 |

Only one task scored below 1.0 on any judge dimension:
**mixed-answerable** ("What's 2 plus 2, and also, what's the weather like
today?") scored `output_quality=0.5` — the judge's rationale: the agent
correctly avoided fabricating a weather answer it had no tool for, but left
half the request genuinely unaddressed. That is a real, defensible
distinction the judge draws — not a scoring bug — see the full report.

## A rubric bug found and fixed mid-run

The first run scored `capital-distractor` ("What is the capital of
France?", answered directly with no tool call — the correct behavior) at
`tool_selection=0.0` and `tool_execution=0.0`. The judge's own rationale
explained why: *"there was no tool usage to evaluate."* The rubric prompt
never told the judge how to score a task that needed zero tools, so it
defaulted to treating "no tool call" as an automatic failure rather than
the ideal outcome.

Fixed by adding one sentence to each dimension's guidance in
`agent_eval/judge.py` (*"if the task needed no tool ... score it 1.0, not
0.0"*) and re-running — `capital-distractor` moved to a clean 1.0/1.0. See
the full report for why this is worth flagging rather than quietly fixing.
