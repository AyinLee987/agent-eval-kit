import copy
import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.intervention_ladder import analyze


def row(task="t1", trial=0, score=True, **changes):
    item = {"task_id": task, "trial": trial, "exact_sequence_match": score,
            "step_recall": float(score), "answer_ok": score, "failure_kind": "match" if score else "skip",
            "expect_tool_sequence": ["first", "second"], "steps": 3}
    item.update(changes)
    return item


def dump(path, conditions, *, task_count=2, repeat=2):
    blob = {"task_count": task_count, "repeat": repeat, "conditions": {
        name: {"condition": name, "total": len(rows), "rows": rows}
        for name, rows in conditions.items()
    }}
    path.write_text(json.dumps(blob), encoding="utf-8")
    return str(path)


def complete():
    return [row(task, trial) for task in ("t1", "t2") for trial in (0, 1)]


def test_load_retains_error_conditions_and_requires_explicit_duplicate_source(tmp_path):
    errored = complete()
    errored[0]["error"] = "API failed"
    first = dump(tmp_path / "first.json", {"baseline": complete(), "instruction": errored})
    second = dump(tmp_path / "second.json", {"instruction": complete()})
    loaded = analyze.load([first])
    assert "instruction" in loaded
    assert loaded["instruction"]["rows"][0]["error"] == "API failed"
    with pytest.raises(analyze.AnalysisError, match="Duplicate condition"):
        analyze.load([first, second])
    chosen = analyze.load([first, second], condition_sources={"instruction": second})
    assert not any(r.get("error") for r in chosen["instruction"]["rows"])
    assert chosen["instruction"]["_excluded_sources"] == [first]


def test_api_error_requires_opt_in_and_excludes_whole_task_not_condition(tmp_path):
    changed = complete()
    changed[0]["error"] = "API failed"
    loaded = analyze.load([dump(tmp_path / "data.json", {"baseline": complete(), "instruction": changed})])
    with pytest.raises(analyze.ComparisonError, match="Incomplete"):
        analyze.analyze_conditions(loaded)
    report = analyze.analyze_conditions(loaded, allow_incomplete=True)
    assert report["conditions"]["instruction"]["api_errors"] == 1
    paired = report["comparisons"]["exact_sequence_match"]["instruction"]
    assert paired["arms"]["b"]["execution_errors"] == 1
    assert paired["paired"]["n_units"] == 1
    assert paired["excluded_units"][0]["task_ids"] == ["t1"]
    assert paired["paired"]["ci"] is paired["paired"]["test"] is None
    assert "instruction | 4 | 4 | 1 | 0" in analyze.render(report)


def test_missing_known_trial_is_not_averaged_with_fewer_runs(tmp_path):
    missing = complete()[1:]
    loaded = analyze.load([dump(tmp_path / "data.json", {"baseline": complete(), "instruction": missing})])
    with pytest.raises(analyze.ComparisonError, match="Incomplete"):
        analyze.analyze_conditions(loaded)
    report = analyze.analyze_conditions(loaded, allow_incomplete=True)
    assert report["conditions"]["instruction"]["missing_records"] == 1
    assert report["comparisons"]["step_recall"]["instruction"]["paired"]["n_units"] == 1


def test_missing_trial_identity_is_never_reconstructed_from_row_order(tmp_path):
    rows = complete()
    del rows[0]["trial"]
    loaded = analyze.load([dump(tmp_path / "data.json", {"baseline": rows})])
    with pytest.raises(analyze.AnalysisError, match="actual task_id"):
        analyze.analyze_conditions(loaded, allow_incomplete=True)


def test_duplicate_trial_is_rejected_even_if_row_count_matches(tmp_path):
    rows = complete()
    rows[1] = copy.deepcopy(rows[0])
    loaded = analyze.load([dump(tmp_path / "data.json", {"baseline": rows})])
    with pytest.raises(analyze.AnalysisError, match="Duplicate task/trial"):
        analyze.analyze_conditions(loaded, allow_incomplete=True)


def test_trial_missing_everywhere_cannot_be_invented_from_repeat_count(tmp_path):
    rows = [row("t1", 7), row("t2", 7)]
    loaded = analyze.load([dump(tmp_path / "data.json", {"baseline": rows})])
    with pytest.raises(analyze.AnalysisError, match="cannot reconstruct") as error:
        analyze.analyze_conditions(loaded, allow_incomplete=True)
    assert error.value.diagnostics[0]["observed_trial_ids"] == [7]


def test_real_nonzero_trial_identifiers_are_preserved(tmp_path):
    rows = [row(task, trial) for task in ("t1", "t2") for trial in (7, 9)]
    report = analyze.analyze_conditions(analyze.load([dump(tmp_path / "data.json", {"baseline": rows})]))
    assert report["observed_trial_ids"] == [7, 9]


def test_changed_expected_tool_sequence_for_same_id_is_rejected(tmp_path):
    changed = complete()
    changed[0]["expect_tool_sequence"] = ["different"]
    loaded = analyze.load([dump(tmp_path / "data.json", {"baseline": complete(), "instruction": changed})])
    with pytest.raises(analyze.AnalysisError, match="changed"):
        analyze.analyze_conditions(loaded)


def test_no_legacy_fingerprints_means_no_ci_or_p_even_with_full_data(tmp_path):
    report = analyze.analyze_conditions(analyze.load([dump(tmp_path / "data.json", {"baseline": complete(), "instruction": complete()})]))
    assert report["legacy_unverified_lineage"] is True
    for comparisons in report["comparisons"].values():
        for comparison in comparisons.values():
            assert comparison["paired"]["n_units"] == 2
            assert comparison["paired"]["test"] is comparison["paired"]["ci"] is None
    assert "CI/p disabled" in analyze.render(report)
    json.dumps(report, allow_nan=False)


def test_missing_cost_is_counted_not_imputed_as_zero(tmp_path):
    report = analyze.analyze_conditions(analyze.load([dump(tmp_path / "data.json", {"baseline": complete()})]))
    assert report["conditions"]["baseline"]["costs"]["delegate_calls"] == {"observed": 0, "missing": 4, "mean_observed": None}


def test_cli_reports_duplicates_and_supports_explicit_selection(tmp_path, capsys):
    first = dump(tmp_path / "first.json", {"baseline": complete()})
    second = dump(tmp_path / "second.json", {"baseline": complete()})
    assert analyze.main([first, second]) == 2
    assert "--condition-source" in capsys.readouterr().err
    out = tmp_path / "report.json"
    assert analyze.main([first, second, "--condition-source", f"baseline={second}", "--json-out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["legacy_unverified_lineage"] is True


def test_duplicate_json_object_key_rejected(tmp_path):
    source = tmp_path / "duplicate.json"
    source.write_text('{"conditions":{"baseline":{},"baseline":{}}}', encoding="utf-8")
    with pytest.raises(analyze.AnalysisError, match="Duplicate JSON"):
        analyze.load([str(source)])


def test_historical_four_file_analysis_never_modifies_results_or_claims_significance():
    root = Path(__file__).resolve().parents[1] / "benchmarks/intervention_ladder"
    paths = [root / name for name in ("results_flat_baseline.json", "results_flat_prompts.json", "results_hier.json", "results_hier_ablation.json")]
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    with pytest.raises(analyze.AnalysisError, match="Duplicate condition"):
        analyze.load([str(path) for path in paths])
    conditions = analyze.load([str(path) for path in paths], condition_sources={"instruction": str(paths[1]), "fewshot": str(paths[1])})
    report = analyze.analyze_conditions(conditions)
    assert len(report["conditions"]) == 6
    assert report["planned_tasks"] == 60
    assert all(comparison["paired"]["n_units"] == 60 and comparison["paired"]["test"] is None for comparisons in report["comparisons"].values() for comparison in comparisons.values())
    assert hashes == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
