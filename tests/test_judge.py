import json

import pytest

from agent_eval.judge import build_judge_prompt, build_llm_judge_fn, render_trajectory
from agent_eval.scoring import TrajectoryJudgeScorer
from agent_eval.types import AgentOutcome, ToolCall, TrajectoryStep


def _outcome() -> AgentOutcome:
    return AgentOutcome(
        answer="391",
        success=True,
        stop_reason="finished",
        steps=1,
        tokens=12,
        trajectory=[
            TrajectoryStep(
                thought="Multiply the two numbers.",
                action=ToolCall(name="calculator", arguments={"expression": "23*17"}),
                observation="391",
            )
        ],
    )


def test_render_trajectory_includes_thought_action_and_observation():
    rendered = render_trajectory(_outcome())
    assert "calculator({'expression': '23*17'})" in rendered
    assert "Multiply the two numbers." in rendered
    assert "391" in rendered


def test_build_judge_prompt_includes_task_and_all_dimensions():
    messages = build_judge_prompt("What is 23 times 17?", _outcome(), dimensions=("tool_selection", "output_quality"))
    user_content = messages[-1]["content"]
    assert "What is 23 times 17?" in user_content
    assert "tool_selection" in user_content
    assert "output_quality" in user_content
    assert "process_control" not in user_content  # only requested dimensions appear


def test_build_llm_judge_fn_parses_json_reply():
    fake_reply = json.dumps({
        "tool_selection": 1.0,
        "output_quality": 0.8,
        "rationale": "Used the calculator correctly and reported the right answer.",
    })
    judge_fn = build_llm_judge_fn(lambda messages: fake_reply, dimensions=("tool_selection", "output_quality"))

    result = judge_fn({"prompt": "What is 23 times 17?"}, _outcome())

    assert result["tool_selection"] == 1.0
    assert result["output_quality"] == 0.8
    assert "correctly" in result["rationale"]


def test_build_llm_judge_fn_handles_a_markdown_fenced_json_reply():
    fenced = "Here is my evaluation:\n```json\n{\"tool_selection\": 0.5, \"rationale\": \"ok\"}\n```"
    judge_fn = build_llm_judge_fn(lambda messages: fenced, dimensions=("tool_selection",))

    result = judge_fn({"prompt": "x"}, _outcome())

    assert result["tool_selection"] == 0.5


def test_build_llm_judge_fn_raises_on_a_reply_with_no_json():
    judge_fn = build_llm_judge_fn(lambda messages: "I refuse to answer in JSON.")

    with pytest.raises(ValueError):
        judge_fn({"prompt": "x"}, _outcome())


def test_trajectory_judge_scorer_namespaces_dimension_keys_and_keeps_rationale():
    judge_fn = lambda task, outcome: {"tool_selection": 1.0, "output_quality": 0.9, "rationale": "solid run"}
    scorer = TrajectoryJudgeScorer(judge_fn)

    scores = scorer.score({"id": "t1"}, _outcome())

    assert scores == {
        "judge_tool_selection": 1.0,
        "judge_output_quality": 0.9,
        "judge_rationale": "solid run",
    }
