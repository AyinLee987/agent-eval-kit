"""Public data provenance, split integrity, label separation and deterministic scoring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval.dataset_scoring import (
    PublicDatasetScorer, gsm8k_numeric_score, load_public_tasks, public_case_prompt,
    qa_answer_scores, score_public_case, supporting_fact_scores,
)
from agent_eval.types import AgentOutcome
from benchmarks.public_data.prepare import (
    HERE, _download, normalize_gsm8k, normalize_hotpotqa, question_family,
    select_splits, sha256, verify,
)


def _qa_case():
    return {
        "schema_version": 1, "id": "hotpotqa:fixture", "source_id": "fixture",
        "dataset": "hotpotqa", "source_split": "validation", "split": "dev",
        "family_id": "family-fixture", "question": "What is the answer?",
        "answer": "Hidden Gold Label",
        "context": [
            {"title": "First document", "sentences": ["One supplied passage."]},
            {"title": "Second document", "sentences": ["A second supplied passage."]},
        ],
        "supporting_facts": [["First document", 0], ["Second document", 0]],
        "metadata": {"private_annotation": "DO NOT PUT LABELS IN THE PROMPT"},
    }


def test_gsm8k_normalization_preserves_source_identity_and_separates_final_answer():
    raw = [{"question": "Rita buys 3 apples and 4 apples. How many apples?",
            "answer": "Add the apples: <<3+4=7>>7.\n#### 7"}]
    case = normalize_gsm8k(raw, "train")[0]
    assert case["source_id"] == "train:0"
    assert case["metadata"]["source_row_index"] == 0
    assert case["answer"] == "7"
    assert case["reference_solution"] == raw[0]["answer"]
    assert case["reference_solution"] not in public_case_prompt(case)
    assert question_family(raw[0]["question"]) == question_family(
        "Rita buys 30 apples and 40 apples. How many apples?"
    )


def test_hotpot_normalization_retains_two_hop_evidence_and_reports_invalid_rows():
    valid = {
        "id": "original-id", "question": "A multihop question?", "answer": "answer",
        "context": {"title": ["First", "Second"], "sentences": [["one"], ["two"]]},
        "supporting_facts": {"title": ["First", "Second"], "sent_id": [0, 0]},
        "type": "bridge", "level": "hard",
    }
    invalid = {**valid, "id": "invalid", "supporting_facts": {
        "title": ["First", "Second"], "sent_id": [0, 20],
    }}
    rows, rejected = normalize_hotpotqa([valid, invalid])
    assert len(rows) == 1
    assert rows[0]["source_id"] == "original-id"
    assert rows[0]["supporting_facts"] == [["First", 0], ["Second", 0]]
    assert rejected == {"support_outside_provided_context": 1}


def test_eval_native_loader_keeps_gold_and_annotations_outside_the_prompt(tmp_path):
    case = _qa_case()
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(case) + "\n", encoding="utf-8")
    task = load_public_tasks(str(path))[0]
    assert task["id"] == case["id"]
    assert task["group_id"] == "hotpotqa:family-fixture"
    assert task["public_case"] == case
    assert case["answer"] not in task["prompt"]
    assert case["metadata"]["private_annotation"] not in task["prompt"]
    assert "[0] One supplied passage." in task["prompt"]
    assert "supporting_facts" in task["prompt"]


@pytest.mark.parametrize("prediction,expected,score", [
    ("42", "42", 1), ("42.0", "42", 1), ("4.2e1", "42", 1),
    ("1,000", "1000", 1), ("−3", "-3", 1), (".5", "0.5", 1),
    ("I considered 42, but the answer is 24.", "42", 0),
    ("Work used 42.\n#### 24", "42", 0),
    ("Work used 24.\n#### 42", "42", 1),
    ("NaN", "42", 0), ("42 or 43", "42", 0),
])
def test_numeric_scoring_uses_the_explicit_final_answer(prediction, expected, score):
    assert gsm8k_numeric_score(prediction, expected) == score


def test_format_adherence_is_separate_from_numeric_correctness():
    result = score_public_case({"dataset": "gsm8k", "answer": "42"}, "Explanation\n#### 42")
    assert result["public_numeric_exact"] == 1
    assert result["public_format_valid"] == 0
    assert result["public_answer_em"] is None


def test_qa_answer_metrics_follow_hotpot_normalization_and_yes_no_rules():
    assert qa_answer_scores("The Eiffel Tower.", "Eiffel Tower")["em"] == 1
    assert qa_answer_scores("red red blue", "red blue")["precision"] == pytest.approx(2 / 3)
    assert qa_answer_scores("yes probably", "yes")["f1"] == 0
    assert qa_answer_scores("no", "yes")["f1"] == 0


def test_supporting_fact_duplicates_do_not_inflate_scores():
    result = supporting_fact_scores(
        [["First", 0], ["First", 0]], [["First", 0], ["Second", 0]],
    )
    assert result["precision"] == 1
    assert result["recall"] == 0.5
    assert result["f1"] == pytest.approx(2 / 3)


def test_qa_scoring_keeps_answer_evidence_and_format_distinct():
    case = _qa_case()
    prediction = json.dumps({"answer": case["answer"], "supporting_facts": [["First document", 0]]})
    result = score_public_case(case, prediction)
    assert result["public_answer_em"] == 1
    assert result["public_evidence_recall"] == 0.5
    assert result["public_joint_em"] == 0
    assert result["public_joint_f1"] == pytest.approx(2 / 3)
    assert result["public_numeric_exact"] is None


@pytest.mark.parametrize("prediction", [
    "not JSON",
    '{"answer":"Hidden Gold Label","answer":"other","supporting_facts":[]}',
    '{"answer":"Hidden Gold Label","supporting_facts":[["First document",true]]}',
    '{"answer":"Hidden Gold Label","supporting_facts":[["Unknown document",0]]}',
    '{"answer":"Hidden Gold Label","supporting_facts":NaN}',
])
def test_invalid_qa_output_never_passes_the_format_metric(prediction):
    assert score_public_case(_qa_case(), prediction)["public_format_valid"] == 0


def test_public_scorer_uses_the_harness_contract_and_its_own_metric_namespace():
    case = _qa_case()
    outcome = AgentOutcome(
        answer=json.dumps({"answer": case["answer"], "supporting_facts": case["supporting_facts"]}),
        success=True, stop_reason="finished", steps=1, tokens=10,
    )
    scorer = PublicDatasetScorer()
    scores = scorer.score({"public_case": case}, outcome)
    assert all(name.startswith("public_") for name in scorer.metric_names)
    assert set(scores) == set(scorer.metric_names)
    assert scores["public_joint_em"] == 1
    assert scores["public_format_valid"] == 1


def _split_case(identifier, *, family=None, context=None):
    return {
        "dataset": "hotpotqa", "source_id": identifier, "id": f"fixture:{identifier}",
        "family_id": family or identifier, "question": f"Question about {identifier}?",
        "context": [{"title": context or identifier, "sentences": ["A sentence."]}],
    }


def test_split_sampling_is_order_independent_and_excludes_shared_test_evidence():
    frozen = _split_case("frozen", family="family", context="Shared article")
    pool = [
        _split_case("same-family", family="family"),
        _split_case("same-context", context="Shared article"),
        _split_case("alpha"), _split_case("beta"), _split_case("gamma"),
    ]
    first = select_splits(pool, [frozen], seed=42, dev_size=2, test_size=1)
    second = select_splits(list(reversed(pool)), [frozen], seed=42, dev_size=2, test_size=1)
    assert first == second
    assert {row["source_id"] for row in first[0]} <= {"alpha", "beta", "gamma"}
    assert all(first[2][key] == 0 for key in (
        "overlapping_family_count", "overlapping_masked_question_count", "overlapping_context_title_count",
    ))


def test_offline_source_validation_cannot_silently_redownload_bad_cache(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline mode must not access the network.")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    source = {"name": "raw.json", "sha256": sha256(b"correct"), "url": "https://example.invalid/raw.json"}
    (tmp_path / source["name"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mismatched cached source"):
        _download(source, tmp_path, offline=True)


def test_published_public_data_has_1600_verified_real_cases_and_separate_splits():
    counts = verify(HERE)
    assert counts == {
        "gsm8k/dev.jsonl": 400, "gsm8k/test.jsonl": 400,
        "hotpotqa/dev.jsonl": 400, "hotpotqa/test.jsonl": 400,
    }
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["datasets"]["gsm8k"]["source_count"] == {"train": 7473, "test": 1319}
    assert manifest["datasets"]["hotpotqa"]["source_count"] == {"validation": 7405}
    assert manifest["datasets"]["hotpotqa"]["license"] == "CC-BY-SA-4.0"
    assert "NOT the official hidden test" in manifest["datasets"]["hotpotqa"]["split_strategy"]
    assert all(source["sha256"] and source["revision"] and source["downloaded_at"] for source in manifest["sources"])


def test_published_gold_labels_roundtrip_through_the_deterministic_scorer():
    for dataset in ("gsm8k", "hotpotqa"):
        for split in ("dev", "test"):
            tasks = load_public_tasks(str(HERE / dataset / f"{split}.jsonl"))
            for task in tasks:
                case = task["public_case"]
                prediction = case["answer"] if dataset == "gsm8k" else json.dumps({
                    "answer": case["answer"], "supporting_facts": case["supporting_facts"],
                })
                result = score_public_case(case, prediction)
                assert result["public_format_valid"] == 1, task["id"]
                metric = "public_numeric_exact" if dataset == "gsm8k" else "public_joint_em"
                assert result[metric] == 1, task["id"]


def test_public_scorer_is_not_applicable_to_native_tasks_but_rejects_malformed_payloads():
    scorer = PublicDatasetScorer()
    outcome = AgentOutcome(answer="42", success=True, stop_reason="finished", steps=1, tokens=1)
    assert scorer.score({"id": "native", "prompt": "Compute."}, outcome) == {
        metric: None for metric in scorer.metric_names
    }
    with pytest.raises(ValueError, match="public_case"):
        scorer.score({"public_case": None}, outcome)
