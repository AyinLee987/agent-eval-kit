"""Load public benchmark tasks without exposing labels, and score them offline.

The HotpotQA normalization and metric definitions follow its official evaluator.
Dataset licenses and attribution live in benchmarks/public_data/licenses.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import AgentOutcome

METRIC_NAMES = (
    "public_format_valid", "public_numeric_exact",
    "public_answer_em", "public_answer_f1",
    "public_evidence_em", "public_evidence_precision",
    "public_evidence_recall", "public_evidence_f1",
    "public_joint_em", "public_joint_f1",
)
_NUMERIC = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?$")


def _numeric_answer(text: str) -> Optional[Decimal]:
    value = text.strip().replace("−", "-")
    if "####" in value:
        value = value.rsplit("####", 1)[1].strip()
    if not _NUMERIC.fullmatch(value):
        return None
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def gsm8k_numeric_score(prediction: str, answer: str) -> float:
    """Compare an explicit final numeric answer, never an intermediate number."""

    expected = _numeric_answer(answer)
    if expected is None:
        raise ValueError("GSM8K gold answer is not a finite number.")
    actual = _numeric_answer(prediction)
    return float(actual is not None and actual == expected)


def _normalize_qa(text: str) -> str:
    lowered = text.lower()
    unpunctuated = "".join(character for character in lowered if character not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", unpunctuated).split())


def qa_answer_scores(prediction: str, answer: str) -> Dict[str, float]:
    """HotpotQA answer EM and token F1, including its yes/no/noanswer rule."""

    predicted, expected = _normalize_qa(prediction), _normalize_qa(answer)
    exact = float(predicted == expected)
    special = {"yes", "no", "noanswer"}
    if predicted != expected and (predicted in special or expected in special):
        return {"em": exact, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    predicted_tokens, expected_tokens = predicted.split(), expected.split()
    overlap = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
    precision = overlap / len(predicted_tokens) if predicted_tokens else 0.0
    recall = overlap / len(expected_tokens) if expected_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"em": exact, "f1": f1, "precision": precision, "recall": recall}


def _fact_pairs(facts: Sequence[Sequence[Any]]) -> set:
    pairs = set()
    for fact in facts:
        if (not isinstance(fact, (list, tuple)) or len(fact) != 2
                or not isinstance(fact[0], str) or not fact[0]
                or isinstance(fact[1], bool) or not isinstance(fact[1], int) or fact[1] < 0):
            raise ValueError("Supporting facts must be [document title, nonnegative sentence index] pairs.")
        pairs.add((fact[0], fact[1]))
    return pairs


def supporting_fact_scores(
    prediction: Sequence[Sequence[Any]], expected: Sequence[Sequence[Any]],
) -> Dict[str, float]:
    """Score sets of title/sentence pairs; duplicate predictions earn no credit."""

    predicted, gold = _fact_pairs(prediction), _fact_pairs(expected)
    overlap = len(predicted & gold)
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(gold) if gold else 0.0
    return {
        "em": float(predicted == gold), "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _qa_prediction(prediction: str) -> Optional[Dict[str, Any]]:
    def unique_fields(pairs):
        fields = {}
        for key, value in pairs:
            if key in fields:
                raise ValueError("Duplicate prediction key.")
            fields[key] = value
        return fields

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON number {value}.")

    try:
        value = json.loads(
            prediction, object_pairs_hook=unique_fields, parse_constant=reject_constant,
        )
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def score_public_case(case: Dict[str, Any], prediction: str) -> Dict[str, Optional[float]]:
    """Produce independent answer, evidence and output-format metrics."""

    scores: Dict[str, Optional[float]] = {name: None for name in METRIC_NAMES}
    if case["dataset"] == "gsm8k":
        scores["public_format_valid"] = float(
            _numeric_answer(prediction) is not None and "####" not in prediction
        )
        scores["public_numeric_exact"] = gsm8k_numeric_score(prediction, case["answer"])
        return scores
    if case["dataset"] != "hotpotqa":
        raise ValueError(f"Unsupported public dataset {case['dataset']!r}.")

    parsed = _qa_prediction(prediction)
    answer = parsed.get("answer") if parsed is not None else None
    facts = parsed.get("supporting_facts") if parsed is not None else None
    answer_valid = isinstance(answer, str) and bool(answer.strip())
    facts_valid = isinstance(facts, list)
    if facts_valid:
        try:
            predicted_pairs = _fact_pairs(facts)
        except ValueError:
            facts_valid = False
        else:
            available = {
                (document["title"], index)
                for document in case["context"] for index in range(len(document["sentences"]))
            }
            facts_valid = predicted_pairs <= available
    scores["public_format_valid"] = float(
        parsed is not None and set(parsed) == {"answer", "supporting_facts"}
        and answer_valid and facts_valid
    )
    answer_scores = qa_answer_scores(answer if answer_valid else "", case["answer"])
    try:
        evidence_scores = supporting_fact_scores(
            facts if isinstance(facts, list) else [], case["supporting_facts"],
        )
    except ValueError:
        evidence_scores = supporting_fact_scores([], case["supporting_facts"])
    scores.update({
        "public_answer_em": answer_scores["em"],
        "public_answer_f1": answer_scores["f1"],
        "public_evidence_em": evidence_scores["em"],
        "public_evidence_precision": evidence_scores["precision"],
        "public_evidence_recall": evidence_scores["recall"],
        "public_evidence_f1": evidence_scores["f1"],
        "public_joint_em": answer_scores["em"] * evidence_scores["em"],
    })
    joint_precision = answer_scores["precision"] * evidence_scores["precision"]
    joint_recall = answer_scores["recall"] * evidence_scores["recall"]
    scores["public_joint_f1"] = (
        2 * joint_precision * joint_recall / (joint_precision + joint_recall)
        if joint_precision + joint_recall else 0.0
    )
    return scores


def public_case_prompt(case: Dict[str, Any]) -> str:
    """Render model input exclusively from question and supplied context."""

    if case["dataset"] == "gsm8k":
        return (
            "Solve this arithmetic word problem. Return only the final number "
            "without units or an explanation.\n\nQuestion: " + case["question"]
        )
    if case["dataset"] != "hotpotqa":
        raise ValueError(f"Unsupported public dataset {case['dataset']!r}.")
    lines = [
        "Answer the question using the supplied passages as evidence.",
        'Return exactly one JSON object: {"answer":"short answer",'
        '"supporting_facts":[["exact document title",0]]}.',
        "Cite every supporting sentence needed for your answer. Sentence indices are zero-based.",
        "", "Question: " + case["question"], "", "Context:",
    ]
    for document_index, document in enumerate(case["context"], start=1):
        lines.append(f"Document {document_index}: {json.dumps(document['title'], ensure_ascii=False)}")
        lines.extend(f"[{index}] {sentence}" for index, sentence in enumerate(document["sentences"]))
        lines.append("")
    return "\n".join(lines)


def load_public_tasks(path: str) -> List[Dict[str, Any]]:
    """Load one normalized JSONL split as eval-native tasks, keeping gold separate."""

    tasks = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            if case["id"] in seen:
                raise ValueError(f"Duplicate public case id on line {line_number}.")
            seen.add(case["id"])
            group_id = f"{case['dataset']}:{case['family_id']}"
            tasks.append({
                "id": case["id"], "prompt": public_case_prompt(case),
                "group_id": group_id, "family_id": case["family_id"],
                "public_case": case,
            })
    return tasks


class PublicDatasetScorer:
    """Deterministic public benchmark scoring compatible with the eval harness."""

    name = "public_dataset"
    metric_names = METRIC_NAMES

    def score(self, task: Dict[str, Any], outcome: AgentOutcome) -> Dict[str, Optional[float]]:
        if "public_case" not in task:
            return {name: None for name in self.metric_names}
        case = task["public_case"]
        if not isinstance(case, dict):
            raise ValueError("PublicDatasetScorer requires task.public_case.")
        return score_public_case(case, outcome.answer)
