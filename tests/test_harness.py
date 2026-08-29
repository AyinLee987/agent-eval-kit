import os

from adapters.bare_baseline import adapt, build_agent
from agent_eval.harness import EvalHarness
from agent_eval.scoring import RuleScorer, ToolUsageScorer, TrajectoryScorer

TASKS_PATH = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "tasks.json")


def _harness() -> EvalHarness:
    return EvalHarness(
        build_agent=build_agent,
        outcome_adapter=adapt,
        tasks=TASKS_PATH,
        scorers=[RuleScorer(), ToolUsageScorer(), TrajectoryScorer()],
    )


def test_bare_baseline_passes_every_shipped_sample_task():
    scorecard = _harness().run_all()

    assert scorecard.total == 4
    agg = scorecard.aggregate()
    assert agg["rule_pass"] == 1.0
    assert agg["trajectory_score"] == 1.0


def test_render_pads_none_valued_cells_so_columns_stay_aligned():
    scorecard = _harness().run_all()
    rendered = scorecard.render()

    lines = [line for line in rendered.splitlines() if line.startswith("echo")]
    assert len(lines) == 1
    # "echo" has no expect_tool, so used_expected_tool is None for it — the
    # "-" placeholder must be padded like every numeric cell, or later
    # columns (trajectory_score here) shift left and misalign.
    assert "-         1.00" in lines[0]


def test_scorecard_renders_and_dumps(tmp_path):
    scorecard = _harness().run_all()

    rendered = scorecard.render()
    assert "EVAL SCORECARD" in rendered

    dump_path = tmp_path / "results.json"
    scorecard.dump(str(dump_path))
    assert dump_path.exists()


def test_inline_task_list_bypasses_the_json_file():
    tasks = [{"id": "echo-only", "prompt": "hi", "expect_substrings": ["hi"]}]
    harness = EvalHarness(
        build_agent=build_agent,
        outcome_adapter=adapt,
        tasks=tasks,
        scorers=[RuleScorer()],
    )
    scorecard = harness.run_all()
    assert scorecard.total == 1
    assert scorecard.results[0].scores == {"rule_pass": True}
