# Conversational ability — results

Run with `python benchmarks/conversational/run_benchmark.py`. Full
per-turn and per-conversation scores + judge rationale are in
`results.json`; the full write-up (including a second rubric bug found and
fixed) is `../../reports/conversational-evaluation.md`.

## Setup

- **Agent**: real `ReActAgent`, LLM = Bailian `qwen-plus`, same 3 tools as
  the trajectory benchmark (`calculator`, `lookup_fact`, `current_datetime`).
- **Judge**: DeepSeek (`deepseek-chat`) — different vendor/model than the
  agent, same reasoning as the trajectory benchmark.
- **7 conversations** (`conversations.json`, in Chinese): knowledge
  retention, a mid-conversation correction (does the agent use the
  *updated* fact, not the original one), three requests spread across
  three turns (completeness), two multi-turn tool-chaining conversations,
  and two controls (a single-turn sanity check, a small-talk exchange with
  no tools at all).
- **Architecture note**: `ReActAgent.run()` creates a fresh execution
  context on every call — it does not remember a prior call by itself. Multi-turn
  state is threaded through the new `ConversationHistoryProvider`
  (`adapters/conversation_history.py`), which implements the sibling
  repo's existing `ContextProvider` hook and replays accumulated turns
  before each new one — no changes to the sibling repo were needed.
- **Scoring**: per-turn `RuleScorer` + `ToolUsageScorer` + the new
  `AnswerRelevancyScorer`; per-conversation the new `ConversationJudgeScorer`
  (knowledge retention + conversation completeness, one combined judge call
  per conversation).

## Results

| Metric | Score |
|---|---|
| judge_knowledge_retention (n=7 conversations) | 1.00 |
| judge_conversation_completeness (n=7 conversations) | 1.00 |
| rule_pass (n=16 turns) | 1.00 |
| used_expected_tool (n=16 turns) | 1.00 |
| answer_relevancy (n=16 turns) | 0.98 |

Every conversation scored a clean 1.0/1.0 on knowledge retention and
completeness — including `retention-update-overwrite`, where the agent
correctly reported the *corrected* project name (`api-rate-limits`) rather
than the one stated first, and `completeness-three-asks`, where all three
requests spread across three turns were addressed by the end.

## A second rubric bug, same class as the trajectory benchmark's

The first run of `AnswerRelevancyScorer` scored two purely informational
turns (e.g. *"我叫小明，正在负责 incident-response 项目的客户对接"* — a
self-introduction, not a question) at `answer_relevancy=0.00`, with the
judge explaining: *"the question merely states the user's name and role
without asking anything, so the assistant's offer of help ... does not
address any specific request."* The prompt asked the judge to grade
whether the response "answers the question," but not every conversational
turn *is* a question — this one just shares information, and the agent's
actual behavior (acknowledge it, offer help) was correct.

Fixed the same way as the trajectory benchmark's bug: rewrote
`build_answer_relevancy_prompt` (`agent_eval/judge.py`) to say explicitly
that a statement should be scored on whether the assistant appropriately
acknowledged/used it, not on whether it answered a question that was never
asked. Re-running: `answer_relevancy` moved from **0.83 → 0.98** overall,
and both previously-zeroed turns now score 0.80–1.00 with rationale that
actually engages with what was said. See the full report for why finding
two instances of the same rubric-design mistake in two different scorers
is itself worth noting, not just fixing quietly.
