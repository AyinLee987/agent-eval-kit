"""Agent trajectory evaluation with a cross-model LLM judge.

Runs a real ReActAgent (Bailian / qwen-plus) over a small task set exercising
single-step tool calls, multi-step tool chaining, a recoverable tool error,
a no-tool-needed distractor, and a partially-unanswerable request — then
grades each full trajectory (not just the final answer) with a DIFFERENT
model, DeepSeek, acting as judge, across four rubric dimensions:
tool_selection, tool_execution, process_control, output_quality.

Cross-model judging matters: a model grading its own agent's outputs tends
to score itself more favorably (self-preference bias, a documented
LLM-as-judge failure mode). Bailian/Qwen and DeepSeek are different vendors
with different base models — that's the whole point of not reusing the
agent's own LLM as its judge.

Usage (from the evaluation/ repo root):

    python benchmarks/agent_trajectory/run_benchmark.py
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

from adapters.react_agent_adapter import adapt, build_agent_factory  # noqa: E402
from agent_eval.harness import EvalHarness  # noqa: E402
from agent_eval.judge import build_llm_judge_fn  # noqa: E402
from agent_eval.scoring import (  # noqa: E402
    RuleScorer,
    ToolUsageScorer,
    TrajectoryJudgeScorer,
    TrajectoryScorer,
)

from tools import calculator, current_datetime, lookup_fact  # noqa: E402

TASKS_PATH = BENCHMARK_DIR / "tasks.json"
RESULTS_PATH = BENCHMARK_DIR / "results.json"

AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a calculator, a lookup_fact "
    "tool for known company facts, and a current_datetime tool. Use a tool "
    "only when the task actually needs it; answer directly otherwise. If "
    "part of a request is outside what your tools can determine, say so "
    "plainly rather than guessing."
)


def require_env(name: str) -> None:
    if not os.environ.get(name):
        raise SystemExit(
            f"Refusing to run: {name} is not set (checked {HARNESS_REPO / '.env'})."
        )


def main() -> None:
    require_env("BAILIAN_API_KEY")
    require_env("DEEPSEEK_API_KEY")

    build_agent = build_agent_factory(
        llm_factory=BailianLLM,
        tools_factory=lambda: ToolRegistry([calculator, lookup_fact, current_datetime]),
        system_prompt=AGENT_SYSTEM_PROMPT,
        max_steps=6,
        agent_name="judged-agent",
    )

    judge_llm = DeepSeekLLM()

    def judge_chat_fn(messages):
        return judge_llm.chat(messages, tools=[]).content or ""

    judge_fn = build_llm_judge_fn(judge_chat_fn)

    harness = EvalHarness(
        build_agent=build_agent,
        outcome_adapter=adapt,
        tasks=str(TASKS_PATH),
        scorers=[
            RuleScorer(),
            ToolUsageScorer(),
            TrajectoryScorer(),
            TrajectoryJudgeScorer(judge_fn),
        ],
    )

    print("Agent model: Bailian (qwen-plus)")
    print("Judge model: DeepSeek (deepseek-chat) — a different vendor/model than the agent.\n")

    scorecard = harness.run_all()
    print(scorecard.render())
    scorecard.dump(str(RESULTS_PATH))
    print(f"\nFull results (including per-task judge rationale) written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
