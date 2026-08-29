"""Multi-turn conversational ability evaluation.

Runs a real ReActAgent (Bailian / qwen-plus) through 7 multi-turn
conversations — knowledge retention (including a mid-conversation
correction), cross-turn request completeness, multi-turn tool chaining, and
two single/small-talk controls — then scores:

- each turn individually: RuleScorer (substring + clean finish),
  ToolUsageScorer (where a turn expects a specific tool), and
  AnswerRelevancyScorer (does this turn's answer address what was asked);
- each conversation as a whole: ConversationJudgeScorer, grading knowledge
  retention and cross-turn completeness over the full transcript.

Both LLM-judge scorers use DeepSeek as judge — a different vendor/model
than the Bailian agent being evaluated, to avoid self-preference bias (see
agent_eval/judge.py).

Usage (from the evaluation/ repo root):

    python benchmarks/conversational/run_benchmark.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parents[2]
HARNESS_REPO = Path(os.environ.get("AGENT_HARNESS_PATH", EVAL_ROOT.parent / "agent" / "agent-harness-from-scratch"))
BENCHMARK_DIR = Path(__file__).resolve().parent

for path in (str(EVAL_ROOT), str(HARNESS_REPO), str(BENCHMARK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dotenv import load_dotenv  # noqa: E402

if not load_dotenv(dotenv_path=HARNESS_REPO / ".env"):
    raise SystemExit(f"Could not find a .env file at {HARNESS_REPO / '.env'}.")

from agent import BailianLLM, DeepSeekLLM, ToolRegistry  # noqa: E402

from adapters.react_agent_adapter import adapt, build_agent_and_history_factory  # noqa: E402
from agent_eval.conversation_harness import ConversationHarness  # noqa: E402
from agent_eval.conversation_scoring import ConversationJudgeScorer  # noqa: E402
from agent_eval.judge import build_answer_relevancy_judge_fn, build_conversation_judge_fn  # noqa: E402
from agent_eval.scoring import AnswerRelevancyScorer, RuleScorer, ToolUsageScorer  # noqa: E402

from tools import calculator, current_datetime, lookup_fact  # noqa: E402

CONVERSATIONS_PATH = BENCHMARK_DIR / "conversations.json"
RESULTS_PATH = BENCHMARK_DIR / "results.json"

AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a calculator, a lookup_fact "
    "tool for known company facts, and a current_datetime tool. This is an "
    "ongoing conversation — use anything the user told you earlier in this "
    "same conversation to inform your answers, and if the user corrects or "
    "updates something they said before, use the latest value, not the "
    "earlier one. Use a tool only when the task actually needs it."
)


def require_env(name: str) -> None:
    if not os.environ.get(name):
        raise SystemExit(
            f"Refusing to run: {name} is not set (checked {HARNESS_REPO / '.env'})."
        )


def main() -> None:
    require_env("BAILIAN_API_KEY")
    require_env("DEEPSEEK_API_KEY")

    build_agent_and_history = build_agent_and_history_factory(
        llm_factory=BailianLLM,
        tools_factory=lambda: ToolRegistry([calculator, lookup_fact, current_datetime]),
        system_prompt=AGENT_SYSTEM_PROMPT,
        max_steps=6,
        agent_name="conversational-agent",
    )

    judge_llm = DeepSeekLLM()

    def judge_chat_fn(messages):
        return judge_llm.chat(messages, tools=[]).content or ""

    answer_relevancy_judge_fn = build_answer_relevancy_judge_fn(judge_chat_fn)
    conversation_judge_fn = build_conversation_judge_fn(judge_chat_fn)

    harness = ConversationHarness(
        build_agent_and_history=build_agent_and_history,
        outcome_adapter=adapt,
        conversations=str(CONVERSATIONS_PATH),
        turn_scorers=[
            RuleScorer(),
            ToolUsageScorer(),
            AnswerRelevancyScorer(answer_relevancy_judge_fn),
        ],
        conversation_scorers=[ConversationJudgeScorer(conversation_judge_fn)],
    )

    print("Agent model: Bailian (qwen-plus)")
    print("Judge model: DeepSeek (deepseek-chat) — a different vendor/model than the agent.\n")

    scorecard = harness.run_all()
    print(scorecard.render())
    scorecard.dump(str(RESULTS_PATH))
    print(f"\nFull results (including per-turn/per-conversation judge rationale) written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
