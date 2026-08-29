from agent_eval.scoring import LLMJudgeScorer, RuleScorer, ToolUsageScorer, TrajectoryScorer
from agent_eval.types import AgentOutcome, ToolCall, TrajectoryStep


def _outcome(**overrides) -> AgentOutcome:
    defaults = dict(
        answer="the answer is 42",
        success=True,
        stop_reason="finished",
        steps=1,
        tokens=10,
        trajectory=[
            TrajectoryStep(
                thought="compute it",
                action=ToolCall(name="calculator", arguments={"expression": "6*7"}),
                observation="42",
            )
        ],
    )
    defaults.update(overrides)
    return AgentOutcome(**defaults)


def test_rule_scorer_passes_on_matching_substring_and_clean_finish():
    task = {"expect_substrings": ["42"]}
    assert RuleScorer().score(task, _outcome()) == {"rule_pass": True}


def test_rule_scorer_fails_when_force_stopped_even_with_right_substring():
    task = {"expect_substrings": ["42"]}
    outcome = _outcome(stop_reason="max_steps")
    assert RuleScorer().score(task, outcome) == {"rule_pass": False}


def test_rule_scorer_fails_on_missing_substring():
    task = {"expect_substrings": ["99"]}
    assert RuleScorer().score(task, _outcome()) == {"rule_pass": False}


def test_tool_usage_scorer_is_none_when_task_declares_no_expected_tool():
    assert ToolUsageScorer().score({}, _outcome()) == {"used_expected_tool": None}


def test_tool_usage_scorer_detects_expected_tool():
    task = {"expect_tool": "calculator"}
    assert ToolUsageScorer().score(task, _outcome()) == {"used_expected_tool": True}


def test_tool_usage_scorer_flags_missing_expected_tool():
    task = {"expect_tool": "web_search"}
    assert ToolUsageScorer().score(task, _outcome()) == {"used_expected_tool": False}


def test_trajectory_scorer_averages_three_signals():
    task = {"expect_tool": "calculator"}
    assert TrajectoryScorer().score(task, _outcome()) == {"trajectory_score": 1.0}


def test_trajectory_scorer_penalizes_tool_errors_even_if_answer_looks_right():
    task = {"expect_tool": "calculator"}
    outcome = _outcome(
        trajectory=[
            TrajectoryStep(
                action=ToolCall(name="calculator", arguments={"expression": "6*7"}),
                observation="ERROR: boom",
            )
        ]
    )
    result = TrajectoryScorer().score(task, outcome)
    assert result["trajectory_score"] == 2 / 3


def test_llm_judge_scorer_delegates_to_injected_function():
    calls = []

    def fake_judge(task, outcome):
        calls.append((task["id"], outcome.answer))
        return True

    scorer = LLMJudgeScorer(fake_judge)
    result = scorer.score({"id": "t1"}, _outcome())
    assert result == {"judge_pass": True}
    assert calls == [("t1", "the answer is 42")]
