import json

import pytest

from adapters.conversation_history import ConversationHistoryProvider
from agent_eval.conversation_harness import ConversationHarness, ConversationResult, ConversationScorecard
from agent_eval.harness import EvalHarness, Scorecard, TaskResult
from agent_eval.judge import build_llm_judge_fn
from agent_eval.scoring import RuleScorer, TrajectoryJudgeScorer
from agent_eval.score_reporting import metric_summary
from agent_eval.types import AgentOutcome, ConversationOutcome, Turn


def outcome(answer="ok", success=True):
    return AgentOutcome(answer, success, "finished" if success else "fatal", 1, 2)


TASKS = [{"id": str(i), "prompt": str(i), "expect_substrings": ["ok"]} for i in range(3)]


@pytest.mark.parametrize("stage", ["build", "run", "adapt"])
def test_one_execution_failure_preserves_every_case_and_later_results(stage, tmp_path):
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        if stage == "build" and calls == 2:
            raise RuntimeError("builder failed")
        return object()

    def run(agent, prompt):
        if stage == "run" and prompt == "1":
            raise RuntimeError("tool failed")
        return prompt

    def adapt(prompt):
        if stage == "adapt" and prompt == "1":
            raise TypeError("adapter failed")
        return outcome()

    card = EvalHarness(factory, adapt, TASKS, [RuleScorer()], run=run).run_all()
    assert len(card.results) == 3
    assert card.results[1].execution_error["stage"] == stage
    assert card.results[2].scores["rule_pass"] is True
    detail = card.metric_summary()["rule_pass"]
    assert detail["planned"] == 3 and detail["valid"] == 2
    assert detail["execution_error"] == 1
    assert detail["coverage"] == pytest.approx(2 / 3)
    assert card.execution_summary()["success_rate"] == pytest.approx(2 / 3)
    assert card.execution_summary()["scorer_errors"] == 0
    assert card.execution_summary()["unavailable_scorers"] == 1
    card.dump(tmp_path / "report.json")
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["results"][1]["execution_error"]["stage"] == stage
    assert "execution_error=1" in card.render()


def test_judge_failure_is_not_agent_failure_or_silent_missing_score():
    replies = iter(['{"quality": 1}', '{}', '{"quality": 0}'])
    scorer = TrajectoryJudgeScorer(build_llm_judge_fn(lambda _: next(replies), ("quality",)))
    card = EvalHarness(object, lambda _: outcome(), TASKS, [scorer, RuleScorer()],
                       run=lambda a, p: p).run_all()
    summary = card.metric_summary()["judge_quality"]
    assert summary["valid"] == 2 and summary["failed"] == 1
    assert summary["mean"] == 0.5
    assert card.execution_summary()["successful"] == 3
    assert card.results[1].scores["rule_pass"] is True
    assert card.results[1].scorer_errors[0]["type"] == "JudgeValidationError"


def test_every_judge_failure_still_has_a_metric_denominator():
    scorer = TrajectoryJudgeScorer(build_llm_judge_fn(lambda _: '{}', ("quality",)))
    card = EvalHarness(object, lambda _: outcome(), TASKS, [scorer], run=lambda a, p: p).run_all()
    assert card.aggregate() == {}
    assert card.metric_summary()["judge_quality"]["failed"] == 3
    assert "avg judge_quality: unavailable" in card.render()


def test_not_applicable_scores_do_not_count_as_answer_correct_or_agreement():
    tasks = [{"id": "none", "prompt": "hello"}, TASKS[0]]
    card = EvalHarness(object, lambda _: outcome(), tasks, [RuleScorer()],
                       run=lambda a, p: p).run_all()
    detail = card.metric_summary()["answer_correct"]
    assert detail["planned"] == 2 and detail["valid"] == 1 and detail["not_applicable"] == 1
    card.results[0].scores["judge_pass"] = False
    assert card.judge_rule_agreement() is None


def test_bad_custom_score_and_colliding_keys_cannot_corrupt_aggregation(tmp_path):
    class BrokenScorer:
        name = "broken"
        metric_names = ("bad",)

        def score(self, task, result):
            return {"bad": float("nan")}

    card = EvalHarness(object, lambda _: outcome(), TASKS, [BrokenScorer(), RuleScorer()],
                       run=lambda a, p: p).run_all()
    assert "bad" not in card.aggregate()
    assert card.metric_summary()["bad"]["failed"] == 3
    card.dump(tmp_path / "no-nan.json")
    assert "NaN" not in (tmp_path / "no-nan.json").read_text()
    duplicate = EvalHarness(object, lambda _: outcome(), TASKS, [RuleScorer(), RuleScorer()],
                            run=lambda a, p: p).run_all()
    assert duplicate.metric_summary()["rule_pass"]["failed"] == 3
    assert "duplicate" in duplicate.results[0].scorer_errors[0]["message"]


def test_manual_scorecards_do_not_average_nonfinite_numbers():
    card = Scorecard([TaskResult("x", outcome(), {"bad": float("inf")})])
    assert card.aggregate() == {}
    assert card.metric_summary()["bad"]["failed"] == 1


def test_harness_does_not_swallow_keyboard_interrupt():
    def run(*args):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        EvalHarness(object, lambda _: outcome(), TASKS, [], run=run).run_all()


def test_invalid_task_ids_fail_before_running_any_agent():
    def build():
        pytest.fail("invalid task set started an agent")
    with pytest.raises(ValueError, match="Duplicate"):
        EvalHarness(build, lambda _: outcome(), [TASKS[0], TASKS[0]], []).run_all()


def test_failed_conversation_blocks_dependent_turns_and_continues_next_conversation():
    seen = []

    def run(agent, prompt):
        seen.append(prompt)
        if prompt == "bad":
            raise RuntimeError("turn failed")
        return outcome()

    conversations = [
        {"id": "a", "turns": [{"prompt": p, "expect_substrings": ["ok"]}
                               for p in ["first", "bad", "dependent"]]},
        {"id": "b", "turns": [{"prompt": "independent", "expect_substrings": ["ok"]}]},
    ]
    card = ConversationHarness(lambda: (object(), ConversationHistoryProvider()),
        lambda result: result, conversations, turn_scorers=[RuleScorer()], run=run).run_all()
    assert seen == ["first", "bad", "independent"]
    assert len(card.results[0].turn_records) == 3
    assert card.results[0].turn_records[2].outcome.stop_reason == "blocked"
    detail = card.turn_metric_summary()["rule_pass"]
    assert detail["valid"] == 2 and detail["execution_error"] == 1 and detail["blocked"] == 1
    assert card.execution_summary()["successful_conversations"] == 1


def test_scorer_failure_does_not_break_conversation_history():
    class Scorer:
        name = "fails"
        metric_names = ("quality",)
        def score(self, task, result):
            raise ValueError("bad judge")

    history = ConversationHistoryProvider()
    lengths = []
    def run(agent, prompt):
        lengths.append(len(history.prepare(prompt)))
        return outcome()
    task = {"id": "c", "turns": [{"prompt": "first"}, {"prompt": "second"}]}
    card = ConversationHarness(lambda: (object(), history), lambda x: x, [task],
                               turn_scorers=[Scorer()], run=run).run_all()
    assert lengths == [0, 2]
    assert card.execution_summary()["successful_conversations"] == 1
    assert card.turn_metric_summary()["quality"]["failed"] == 2


@pytest.mark.parametrize("metadata", [{"value": float("nan")}, {1, 2}, object()])
def test_bad_score_metadata_is_isolated_before_report_serialization(metadata, tmp_path):
    class Scorer:
        name = "metadata"
        metric_names = ("quality",)
        def score(self, task, result):
            return {"quality": 1, "metadata": metadata}
    card = EvalHarness(object, lambda _: outcome(), TASKS, [Scorer()],
                       run=lambda a, p: p).run_all()
    assert card.metric_summary()["quality"]["failed"] == 3
    card.dump(tmp_path / "valid.json")


def test_manual_conversation_card_cannot_count_absent_records_as_success():
    failed = ConversationOutcome([Turn("q", outcome(success=False))])
    card = ConversationScorecard([ConversationResult("c", failed)])
    assert card.execution_summary()["planned_turns"] == 1
    assert card.execution_summary()["successful_conversations"] == 0
    empty = ConversationScorecard([ConversationResult("c", ConversationOutcome([]))])
    assert empty.execution_summary()["successful_conversations"] == 0


def test_generic_metrics_do_not_infer_a_range_from_small_observed_values():
    summary = metric_summary([
        ({"latency": 0.1}, {"latency": "valid"}),
        ({"latency": None}, {"latency": "failed"}),
    ])["latency"]
    assert summary["mean"] == 0.1 and summary["failed"] == 1
    assert "score_bounds" not in summary


def test_large_finite_custom_metrics_still_have_a_finite_mean():
    summary = metric_summary([({"large": 1e308}, {}) for _ in range(3)])
    assert summary["large"]["mean"] == pytest.approx(1e308)


def test_dump_validation_leaves_existing_report_intact(tmp_path):
    target = tmp_path / "report.json"
    target.write_text("previous complete report", encoding="utf-8")
    card = Scorecard([TaskResult("x", outcome(), {"quality": float("nan")})])
    with pytest.raises(ValueError):
        card.dump(target)
    assert target.read_text(encoding="utf-8") == "previous complete report"
    assert list(tmp_path.iterdir()) == [target]
