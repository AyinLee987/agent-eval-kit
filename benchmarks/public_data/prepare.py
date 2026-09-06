"""Prepare licensed, reproducible public dev and frozen-test subsets.

Online downloads use only source_lock.json's pinned URLs and SHA-256 values.
--offline rebuilds from verified cached sources. --verify checks the published
normalized files and split boundaries without network or optional dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
import unicodedata
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
DEFAULT_SEED = 20260905
MAX_SOURCE_BYTES = 80_000_000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def question_family(question: str) -> str:
    """Group matching wording after numeric values are masked.

    This is an auditable lexical family rule, not a claim that arbitrary
    semantic paraphrases or training-set contamination can be detected.
    """

    normalized = unicodedata.normalize("NFKC", question).casefold()
    normalized = re.sub(r"[-+]?\d+(?:[,.]\d+)*", " number ", normalized)
    return sha256(_canonical(normalized).encode("utf-8"))


def _download(source: Dict[str, Any], cache: Path, offline: bool) -> Dict[str, Any]:
    path = cache / source["name"]
    receipts_path = cache / "receipts.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    cached = path.exists() and sha256(path.read_bytes()) == source["sha256"]
    if not cached:
        if offline:
            raise ValueError(f"Missing or mismatched cached source: {source['name']}.")
        request = urllib.request.Request(source["url"], headers={"User-Agent": "evaluation-public-data/1.0"})
        partial = path.with_name(path.name + ".part")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as target:
                size = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_SOURCE_BYTES or time.monotonic() - started > 240:
                        raise ValueError("Dataset download exceeded its size or duration limit.")
                    target.write(chunk)
            if sha256(partial.read_bytes()) != source["sha256"]:
                raise ValueError(f"Source hash mismatch: {source['name']}.")
            partial.replace(path)
            receipts[source["name"]] = datetime.now(timezone.utc).isoformat()
        finally:
            if partial.exists():
                partial.unlink()
    if source["name"] not in receipts:
        # The initial verified fetch may have been performed before this script.
        receipts[source["name"]] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    receipts_path.write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
    return {
        **source, "downloaded_at": receipts[source["name"]],
        "bytes": path.stat().st_size,
    }


def normalize_gsm8k(rows: Sequence[Dict[str, str]], source_split: str) -> List[Dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows):
        question, solution = row["question"], row["answer"]
        if not isinstance(question, str) or not question.strip() or "####" not in solution:
            raise ValueError(f"Malformed GSM8K source row {source_split}:{index}.")
        answer = solution.rsplit("####", 1)[1].strip()
        try:
            numeric = Decimal(answer.replace(",", ""))
        except InvalidOperation as exc:
            raise ValueError(f"Non-numeric GSM8K answer at {source_split}:{index}.") from exc
        if not numeric.is_finite():
            raise ValueError("GSM8K gold answer must be finite.")
        source_id = f"{source_split}:{index}"
        normalized.append({
            "schema_version": 1, "id": f"gsm8k:{source_id}",
            "dataset": "gsm8k", "source_split": source_split, "source_id": source_id,
            "family_id": question_family(question), "question": question,
            "answer": answer, "reference_solution": solution,
            "metadata": {
                "source_row_index": index, "source_id_kind": "zero_based_line_in_pinned_upstream_split",
                "reasoning_step_count": len(re.findall(r"<<.*?>>", solution)),
                "license": "MIT",
            },
        })
    return normalized


def normalize_hotpotqa(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    normalized, rejected = [], Counter()
    for row in rows:
        contexts = row["context"]
        facts = row["supporting_facts"]
        if isinstance(contexts, dict):
            contexts = list(zip(contexts["title"], contexts["sentences"]))
        if isinstance(facts, dict):
            facts = list(zip(facts["title"], facts["sent_id"]))
        documents = [{"title": title, "sentences": list(sentences)} for title, sentences in contexts]
        by_title = {document["title"]: document["sentences"] for document in documents}
        if len(by_title) != len(documents):
            rejected["duplicate_context_title"] += 1
            continue
        if not row.get("answer") or not facts:
            rejected["missing_answer_or_support"] += 1
            continue
        if any(
            title not in by_title or not isinstance(index, int) or isinstance(index, bool)
            or index < 0 or index >= len(by_title[title])
            for title, index in facts
        ):
            rejected["support_outside_provided_context"] += 1
            continue
        supporting_titles = sorted({title for title, _ in facts})
        if len(supporting_titles) < 2:
            rejected["fewer_than_two_supporting_documents"] += 1
            continue
        source_id = str(row.get("_id", row.get("id")))
        if not source_id or source_id == "None":
            raise ValueError("HotpotQA source row has no original ID.")
        normalized.append({
            "schema_version": 1, "id": f"hotpotqa:{source_id}",
            "dataset": "hotpotqa", "source_split": "validation",
            "source_id": source_id,
            "family_id": sha256(json.dumps(supporting_titles, ensure_ascii=False).encode("utf-8")),
            "question": row["question"], "answer": row["answer"],
            "context": documents, "supporting_facts": [[title, index] for title, index in facts],
            "metadata": {
                "type": row.get("type"), "level": row.get("level"),
                "license": "CC-BY-SA-4.0",
                "upstream_split": "HotpotQA official dev distractor v1",
            },
        })
    return normalized, dict(rejected)


def _context_titles(case: Dict[str, Any]) -> set:
    return {_canonical(document["title"]) for document in case.get("context", [])}


def select_splits(
    dev_pool: Sequence[Dict[str, Any]], test_pool: Sequence[Dict[str, Any]],
    *, seed: int, dev_size: int, test_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Freeze test first, then select dev without overlapping families/evidence."""

    if isinstance(dev_size, bool) or isinstance(test_size, bool) or dev_size < 1 or test_size < 1:
        raise ValueError("Both split sizes must be positive integers.")

    def ranked(pool, split):
        return sorted(pool, key=lambda case: sha256(
            f"{seed}:{case['dataset']}:{split}:{case['source_id']}".encode("utf-8")
        ))

    test, test_families, test_questions, test_ids = [], set(), set(), set()
    for case in ranked(test_pool, "test"):
        question = question_family(case["question"])
        if case["family_id"] in test_families or question in test_questions:
            continue
        test.append({**case, "split": "test"})
        test_families.add(case["family_id"])
        test_questions.add(question)
        test_ids.add(case["source_id"])
        if len(test) == test_size:
            break
    if len(test) != test_size:
        raise ValueError(f"Only {len(test)} unique test families are available.")
    blocked_titles = set().union(*(_context_titles(case) for case in test))

    dev, dev_families, dev_questions = [], set(), set()
    exclusions = Counter()
    for case in ranked(dev_pool, "dev"):
        question = question_family(case["question"])
        if case["source_id"] in test_ids:
            exclusions["shared_source_id"] += 1
            continue
        if case["family_id"] in test_families or question in test_questions:
            exclusions["shared_test_family"] += 1
            continue
        if _context_titles(case) & blocked_titles:
            exclusions["shared_test_context_title"] += 1
            continue
        if case["family_id"] in dev_families or question in dev_questions:
            exclusions["duplicate_dev_family"] += 1
            continue
        dev.append({**case, "split": "dev"})
        dev_families.add(case["family_id"])
        dev_questions.add(question)
        if len(dev) == dev_size:
            break
    if len(dev) != dev_size:
        raise ValueError(f"Only {len(dev)} dev families remain after test separation; requested {dev_size}.")
    return dev, test, {
        "dev_pool_count": len(dev_pool), "test_pool_count": len(test_pool),
        "examined_dev_exclusions": dict(exclusions),
        "overlapping_family_count": len(dev_families & test_families),
        "overlapping_masked_question_count": len(dev_questions & test_questions),
        "overlapping_context_title_count": len(
            set().union(*(_context_titles(case) for case in dev)) & blocked_titles
        ),
    }


def _jsonl(cases: Sequence[Dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for case in cases
    ).encode("utf-8")


def _file_summary(path: str, data: bytes, cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "path": path, "count": len(cases), "bytes": len(data), "sha256": sha256(data),
        "source_splits": dict(Counter(case["source_split"] for case in cases)),
        "task_types": dict(Counter(case["metadata"].get("type", "numeric_reasoning") for case in cases)),
    }


def prepare(
    *, output_dir: Path = HERE, cache_dir: Path = HERE / ".cache",
    seed: int = DEFAULT_SEED, dev_size: int = 400, test_size: int = 400,
    offline: bool = False, replace_frozen: bool = False,
) -> Dict[str, Any]:
    output_dir, cache_dir = Path(output_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_lock = json.loads((HERE / "source_lock.json").read_text(encoding="utf-8"))
    receipts = [_download(source, cache_dir, offline) for source in source_lock["sources"]]
    train = [json.loads(line) for line in (cache_dir / "gsm8k_train.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [json.loads(line) for line in (cache_dir / "gsm8k_test.jsonl").read_text(encoding="utf-8").splitlines()]
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Preparation requires: pip install -r benchmarks/public_data/requirements.txt") from exc
    qa_raw = parquet.read_table(cache_dir / "hotpotqa_validation.parquet").to_pylist()
    qa_pool, qa_rejected = normalize_hotpotqa(qa_raw)
    pools = {
        "gsm8k": (normalize_gsm8k(train, "train"), normalize_gsm8k(test, "test")),
        "hotpotqa": (qa_pool, qa_pool),
    }
    files, datasets, pending = {}, {}, {}
    for dataset, (dev_pool, test_pool) in pools.items():
        development, frozen, audit = select_splits(
            dev_pool, test_pool, seed=seed, dev_size=dev_size, test_size=test_size,
        )
        for split, cases in (("dev", development), ("test", frozen)):
            relative = f"{dataset}/{split}.jsonl"
            data = _jsonl(cases)
            target = output_dir / relative
            if split == "test" and target.exists() and target.read_bytes() != data and not replace_frozen:
                raise ValueError(f"Frozen test differs: {target}. Use a new output directory or --replace-frozen explicitly.")
            files[relative] = _file_summary(relative, data, cases)
            pending[relative] = data
        datasets[dataset] = {
            "license": "MIT" if dataset == "gsm8k" else "CC-BY-SA-4.0",
            "homepage": "https://github.com/openai/grade-school-math" if dataset == "gsm8k" else "https://hotpotqa.github.io/",
            "paper": "https://arxiv.org/abs/2110.14168" if dataset == "gsm8k" else "https://arxiv.org/abs/1809.09600",
            "split_strategy": (
                "dev sampled from upstream train; frozen test sampled from upstream test"
                if dataset == "gsm8k" else
                "both local splits sampled from the official dev distractor split; this is NOT the official hidden test"
            ),
            "family_rule": (
                "number-masked NFKC/casefolded question wording"
                if dataset == "gsm8k" else
                "supporting-title set, plus no shared context title or number-masked question between published splits"
            ),
            "split_audit": audit,
            "source_count": {"train": len(train), "test": len(test)} if dataset == "gsm8k" else {"validation": len(qa_raw)},
            "rejected_source_records": {} if dataset == "gsm8k" else qa_rejected,
        }
    config = {
        "seed": seed, "dev_per_dataset": dev_size, "test_per_dataset": test_size,
        "algorithm": "ascending SHA256(seed:dataset:local_split:source_id), one example per lexical/support family",
        "test_selection_precedes_dev": True,
    }
    snapshot_id = sha256(json.dumps(
        {"files": files, "sampling": config, "source_hashes": [item["sha256"] for item in receipts]},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    created_at = datetime.now(timezone.utc).isoformat()
    prior_manifest = output_dir / "manifest.json"
    if prior_manifest.exists():
        prior = json.loads(prior_manifest.read_text(encoding="utf-8"))
        if prior.get("snapshot_id") == snapshot_id:
            created_at = prior["prepared_at"]
    manifest = {
        "schema_version": 1, "snapshot_id": snapshot_id, "prepared_at": created_at,
        "preparation_script_sha256": sha256(Path(__file__).read_bytes()),
        "sampling": config, "sources": receipts, "datasets": datasets, "files": files,
        "limitations": [
            "These are public benchmarks and may already be present in model pretraining.",
            "Lexical family detection does not certify absence of arbitrary semantic paraphrases.",
            "HotpotQA local frozen test is drawn from public official dev, not the official hidden test.",
            "No model was run to select examples; no accuracy or performance claim follows from preparation.",
        ],
    }
    for relative, data in pending.items():
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    licenses = output_dir / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache_dir / "gsm8k_LICENSE.txt", licenses / "GSM8K-MIT.txt")
    shutil.copyfile(cache_dir / "CC-BY-SA-4.0.txt", licenses / "HOTPOTQA-CC-BY-SA-4.0.txt")
    for path in sorted(licenses.iterdir()):
        if path.is_file():
            manifest.setdefault("license_files", {})[str(path.relative_to(output_dir)).replace("\\", "/")] = sha256(path.read_bytes())
    prior_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify(output_dir: Path = HERE) -> Dict[str, int]:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    counts = {}
    loaded = {}
    for relative, expected in manifest["files"].items():
        path = output_dir / relative
        data = path.read_bytes()
        if sha256(data) != expected["sha256"]:
            raise ValueError(f"Normalized file hash mismatch: {relative}.")
        cases = [json.loads(line) for line in data.decode("utf-8").splitlines()]
        if len(cases) != expected["count"]:
            raise ValueError(f"Normalized count mismatch: {relative}.")
        if len({case["id"] for case in cases}) != len(cases):
            raise ValueError(f"Duplicate case ID in {relative}.")
        counts[relative], loaded[relative] = len(cases), cases
    for dataset in ("gsm8k", "hotpotqa"):
        development, frozen = loaded[f"{dataset}/dev.jsonl"], loaded[f"{dataset}/test.jsonl"]
        for field in ("id", "source_id", "family_id"):
            if {row[field] for row in development} & {row[field] for row in frozen}:
                raise ValueError(f"Cross-split {field} overlap in {dataset}.")
        if {question_family(row["question"]) for row in development} & {
            question_family(row["question"]) for row in frozen
        }:
            raise ValueError(f"Cross-split lexical question overlap in {dataset}.")
        if set().union(*(_context_titles(row) for row in development)) & set().union(
            *(_context_titles(row) for row in frozen)
        ):
            raise ValueError(f"Cross-split context-title overlap in {dataset}.")
    for relative, expected_hash in manifest.get("license_files", {}).items():
        if sha256((output_dir / relative).read_bytes()) != expected_hash:
            raise ValueError(f"License hash mismatch: {relative}.")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--cache-dir", type=Path, default=HERE / ".cache")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dev-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--replace-frozen", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        prepare(
            output_dir=args.output_dir, cache_dir=args.cache_dir, seed=args.seed,
            dev_size=args.dev_size, test_size=args.test_size, offline=args.offline,
            replace_frozen=args.replace_frozen,
        )
    print(json.dumps(verify(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
