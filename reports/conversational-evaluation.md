# Conversational Ability Evaluation Report

**System under test:** the real `ReActAgent` in `agent-harness-from-scratch`,
running Alibaba Bailian's `qwen-plus`, across multi-turn conversations.
**Toolkit used:** the new multi-turn extension to `agent_eval` —
`types.Turn`/`ConversationOutcome`, `conversation_harness.ConversationHarness`,
`conversation_scoring.ConversationJudgeScorer`, `scoring.AnswerRelevancyScorer`.
**Benchmark code:** `benchmarks/conversational/` (`tools.py`,
`conversations.json`, `run_benchmark.py`). Re-run with
`python benchmarks/conversational/run_benchmark.py`.
**Raw output:** `benchmarks/conversational/results.json`.

---

## 1. Objective

Everything measured so far in this project — RAG recall, trajectory
quality — is a **single-turn** capability: one prompt in, one graded
outcome out. Real usage of a chat agent is a conversation: later turns
depend on earlier ones. This evaluation targets specifically the
capabilities that only exist across turns:

- **Knowledge retention** — does the agent remember, and correctly use,
  information the user already provided, including using an *updated*
  fact instead of an earlier one when the user corrects themselves?
- **Conversation completeness** — if a user spreads several distinct
  requests across several turns, does the agent address all of them by
  the end, not just whichever was asked most recently?

A single-turn harness cannot express either metric — both require seeing
the whole conversation as one unit, and the first requires the agent to
actually observe prior turns as input, not just be asked about them.

## 2. A load-bearing architecture fact

Before designing anything, this evaluation checked a claim that turned out
to be false: that calling `ReActAgent.run()` twice on the same instance
continues a conversation. Reading `agent/trigger/react_loop.py:134`
(`ReActLoop.run()`) shows every call builds a **brand-new**
`ExecutionContext` and adds only the system prompt and that call's single
user message — nothing from a prior call carries over. This is not a bug;
it's a design choice separate from the sibling repo's own policy-controlled
long-term `MemoryManager`.

Consequently, multi-turn evaluation **cannot** be "call `run()` a few
times and see what happens" — it requires the harness itself to manage and
re-inject conversation history. The sibling repo already exposes the right
extension point for this without needing any change to it:

```python
class ContextProvider(Protocol):
    def prepare(self, task: str) -> Sequence[Dict[str, Any]]: ...
```

`ReActAgent` calls every registered `context_providers[i].prepare(task)`
and injects the returned messages before the new user message, on every
`run()` call. `adapters/conversation_history.py`'s `ConversationHistoryProvider`
implements this protocol structurally (no import from the sibling repo
needed to satisfy it) and accumulates turns; `ConversationHarness` calls
`history.append_turn(user_message, answer)` after every turn, before the
next one runs. This is the mechanism the whole evaluation rests on — if it
didn't work, no result below would mean anything.

## 3. Architecture added to `agent_eval`

| Addition | Purpose |
|---|---|
| `types.Turn` / `types.ConversationOutcome` | The multi-turn analog of `AgentOutcome` — an ordered list of turns from one continuous session |
| `conversation_harness.ConversationHarness` | Builds **one agent per conversation** (not per turn — turns must share state), runs turns sequentially, scores at both levels |
| `conversation_harness.ConversationScorecard` | Aggregates turn-level scores (pooled across every turn of every conversation) and conversation-level scores separately |
| `conversation_scoring.ConversationScorer` / `ConversationJudgeScorer` | The conversation-level scorer protocol and its LLM-judge implementation |
| `scoring.AnswerRelevancyScorer` | A new turn-level scorer: does this turn's response engage with what was actually said |
| `judge.build_conversation_judge_fn` / `build_answer_relevancy_judge_fn` | Framework-agnostic judge-prompt builders, following the same `chat_fn`-injection pattern as the trajectory judge |
| `adapters/conversation_history.py` | The `ContextProvider` implementation described in §2 |
| `adapters/react_agent_adapter.build_agent_and_history_factory` | Wires a fresh `ConversationHistoryProvider` into a fresh `ReActAgent`'s `context_providers` per conversation |

## 4. Methodology

### 4.1 Conversations (`conversations.json`, n=7, in Chinese)

| id | turns | tests |
|---|---|---|
| `retention-project-name` | 3 | self-introduction → unrelated calc → recall check |
| `retention-update-overwrite` | 3 | stated fact → correction → recall of the *corrected* value |
| `completeness-three-asks` | 3 | three unrelated requests, one per turn |
| `multiturn-tool-chain-rate-limit` | 2 | lookup a fact, then compute from it in the next turn |
| `multiturn-tool-chain-stipend` | 2 | same shape, different fact/computation |
| `control-single-turn` | 1 | sanity control — no multi-turn dependency at all |
| `control-small-talk` | 2 | no tools, casual conversation, tests relevancy in the absence of any task structure |

Each conversation carries `expected_intentions` — a human-readable list of
what the whole conversation should accomplish by the end — consumed by
`ConversationJudgeScorer`'s completeness dimension.

### 4.2 Scoring

- **Per turn**: `RuleScorer` (substring + clean finish, where a turn
  specifies `expect_substrings`), `ToolUsageScorer` (where a turn specifies
  `expect_tool`), `AnswerRelevancyScorer` (LLM-judge, every turn).
- **Per conversation**: `ConversationJudgeScorer`, one combined judge call
  over the full transcript grading `knowledge_retention` and
  `conversation_completeness`.

Both judge scorers call DeepSeek (`deepseek-chat`) — a different vendor and
base model than the Bailian agent being evaluated, for the same
self-preference-bias reason established in the trajectory report.

## 5. Results

| Metric | Score |
|---|---|
| `judge_knowledge_retention` (n=7 conversations) | 1.00 |
| `judge_conversation_completeness` (n=7 conversations) | 1.00 |
| `rule_pass` (n=16 turns) | 1.00 |
| `used_expected_tool` (n=16 turns) | 1.00 |
| `answer_relevancy` (n=16 turns) | 0.98 |

Every conversation scored a clean 1.0/1.0 on both conversation-level
dimensions. Two results worth calling out specifically:

- **`retention-update-overwrite`**: the agent was told a project name,
  then told a corrected one, then asked which project it's on. It answered
  with the *corrected* name (`api-rate-limits`), not the first one stated —
  the exact failure mode this scenario exists to catch.
- **`completeness-three-asks`**: three unrelated requests (arithmetic, a
  fact lookup, a datetime lookup) spread one-per-turn were all answered by
  the end of the conversation.

## 6. A second rubric bug, the same class as the trajectory benchmark's

The first run scored two purely informational turns — e.g. *"我叫小明，
正在负责 incident-response 项目的客户对接"* (a self-introduction, not a
question) — at `answer_relevancy=0.00`. The judge's own rationale: *"the
question merely states the user's name and role without asking anything,
so the assistant's offer of help ... does not address any specific
request."*

**The prompt asked the judge whether the response "answers the question,"
but this turn wasn't a question.** It was a statement, and the agent's
actual behavior — acknowledge it, offer relevant help — was exactly
correct. The rubric had no instruction for how to score a non-question
turn, so it defaulted to treating "nothing specific was asked" as "nothing
was addressed."

**Fix** (`agent_eval/judge.py`, `build_answer_relevancy_prompt`): relabeled
the prompt from "Question"/"Answer" to "User's message"/"Assistant's
response," and added: *"Not every user turn is a question — a statement
that only shares information ... should be scored on whether the assistant
appropriately acknowledged or used it, not on whether it answered a
question that was never asked."*

Re-running: `answer_relevancy` moved from **0.83 to 0.98** overall, and
both previously-zeroed turns now score 0.80–1.00 with rationale that
engages with what was actually said (e.g. *"The assistant appropriately
acknowledged the user's statement about their project and offered relevant
assistance, fully engaging with the shared information."*).

**Why this is reported rather than just fixed:** this is the *second*
time, in two different judge scorers built for two different benchmarks in
this project, that the same underlying mistake appeared — a rubric
implicitly assuming a "question → answer" shape that doesn't hold for
every input (a task needing no tool, in the trajectory benchmark; a
conversational turn that's a statement, here). That is a pattern, not a
one-off typo: **any LLM-judge rubric built around "did X correctly do Y"
needs an explicit answer for "what if Y wasn't actually required here,"**
or it will silently penalize the correct behavior in that case. Future
judge prompts in this toolkit should be written with that check as a
standard step, not rediscovered per-scorer.

## 7. Threats to validity

- **n=7 conversations / 16 turns** is small — enough to exercise each
  targeted behavior (retention, correction, completeness, tool chaining,
  a no-tool control) at least once, and enough to have surfaced two rubric
  bugs across this project so far, but not enough to claim a stable
  "conversational quality" score.
- **A perfect 1.00/1.00 on both conversation-level dimensions across all 7
  conversations is a ceiling result** — this benchmark's scenarios are not
  adversarial (the correction in `retention-update-overwrite` is stated
  plainly, not buried or contradicted subtly). It demonstrates the
  agent handles straightforward multi-turn dependencies correctly and that
  the harness/judge pipeline works end-to-end; it does not demonstrate
  robustness under harder conversational pressure (ambiguous corrections,
  much longer conversations, conflicting requests). That is a natural next
  step, not a claim this run already makes.
- **One judge model, one run, no repeated sampling** — same caveat as the
  trajectory report.
- **What this evaluation supports:** the toolkit can evaluate genuine
  multi-turn behavior (not just repeated single-turn calls) by threading
  history through the sibling repo's existing extension point, with no
  changes to that repo; on straightforward multi-turn scenarios, this
  agent retains and correctly updates information across turns and
  addresses requests spread across turns. **What it does not support:** a
  claim about conversational robustness under adversarial or long-horizon
  conditions — that is planned as ② robustness and ⑤ business-scenario
  evaluation.

## 8. Reproducibility

```bash
# from the evaluation/ repo root, with BAILIAN_API_KEY and DEEPSEEK_API_KEY
# set in the sibling repo's .env
python benchmarks/conversational/run_benchmark.py
```

Not bit-for-bit deterministic — both the agent and the judge are real LLM
calls.

## Appendix: full conversation set

| id | expected intentions | turns |
|---|---|---|
| retention-project-name | 记住自我介绍信息；满足算术请求；正确回忆项目名称 | (1) 我叫小明，正在负责 incident-response 项目的客户对接。 (2) 顺便帮我算一下 23 乘 17。→ 391 (3) 还记得我在负责哪个项目吗？→ incident-response |
| retention-update-overwrite | 记住更正后的最新项目名称 | (1) 我现在负责的项目是 database-backup-schedule。 (2) 抱歉更正一下，我实际上换到 api-rate-limits 这个项目了。 (3) 我现在负责哪个项目？→ api-rate-limits |
| completeness-three-asks | 三个请求都被回应 | (1) 帮我算一下 12 加 30。→ 42 (2) 用 lookup_fact 查 'pto_days_per_year'。→ 15 (3) 用 current_datetime 查现在时间 |
| multiturn-tool-chain-rate-limit | 查到数值并正确计算 | (1) 查 'standard_rate_limit_rpm'。→ 60 (2) 10 分钟内允许多少次请求？→ 600 |
| multiturn-tool-chain-stipend | 查到数值并正确计算 | (1) 查 'oncall_stipend_usd'。→ 200 (2) 一个季度 3 次轮值总共多少钱？→ 600 |
| control-single-turn | 正确回答加法问题 | (1) 13 加 29 等于多少？→ 42 |
| control-small-talk | 礼貌回应；合理自我介绍 | (1) 你好，今天感觉怎么样？ (2) 能简单介绍一下你自己吗？ |
