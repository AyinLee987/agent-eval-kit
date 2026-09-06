"""Check that a review catalog cannot silently claim incomplete Agent coverage."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.agent_design.catalog import load_catalog, render_markdown, validate_catalog


def test_catalog_contains_ten_unique_specifications_per_direction():
    data = load_catalog()
    assert len(data["cases"]) == 50
    assert len({case["id"] for case in data["cases"]}) == 50
    for category in ("tools", "state", "budget", "context", "a2a"):
        cases = [case for case in data["cases"] if case["category"] == category]
        assert len(cases) == 10
        assert any(case["fault"] is None for case in cases)
        assert all(case["status"] == "specified" for case in cases)


def test_render_is_stable_does_not_mutate_and_matches_checked_in_review():
    data = load_catalog()
    before = copy.deepcopy(data)
    actual = render_markdown(data)
    data["cases"].reverse()
    assert render_markdown(data) == actual
    data["cases"].reverse()
    assert data == before
    expected = Path(__file__).parents[1] / "benchmarks" / "agent_design" / "CASES.md"
    assert actual == expected.read_text(encoding="utf-8")
    assert "没有模型实测分数" in actual


@pytest.mark.parametrize("field", ["id", "title", "prompt", "expected", "metrics", "trace_requirements", "required_capabilities", "fault", "setup", "comparison"])
def test_missing_required_case_field_is_rejected(field):
    data = load_catalog()
    del data["cases"][0][field]
    with pytest.raises(ValueError):
        validate_catalog(data)


@pytest.mark.parametrize("mutate", [
    lambda data: data.update(schema_version=True),
    lambda data: data.update(status="passed"),
    lambda data: data["cases"].pop(),
    lambda data: data["cases"][1].update(id=data["cases"][0]["id"]),
    lambda data: data["cases"][0].update(category="skill"),
    lambda data: data["cases"][0].update(id="STATE-01"),
    lambda data: data["cases"][0].update(status="passed"),
    lambda data: data["cases"][0]["setup"]["limits"].update(deadline_seconds=float("nan")),
    lambda data: data["cases"][0]["setup"]["limits"].update(max_total_tokens=-1),
    lambda data: data["cases"][0]["setup"]["limits"].update(max_agent_steps=True),
    lambda data: data["cases"][0].update(trace_requirements=[]),
    lambda data: data["cases"][40]["setup"]["protocol_scope"].update(implementation_status="implemented"),
    lambda data: data["cases"][40].update(required_capabilities=["local_function_delegation"]),
])
def test_invalid_or_misleading_specification_is_rejected(mutate):
    data = load_catalog()
    mutate(data)
    with pytest.raises(ValueError):
        validate_catalog(data)


def test_loading_another_catalog_validates_it_and_does_not_share_mutable_state(tmp_path):
    first = load_catalog()
    source = tmp_path / "cases.json"
    source.write_text(json.dumps(first, ensure_ascii=False), encoding="utf-8")
    loaded = load_catalog(source)
    loaded["cases"][0]["title"] = "改动副本"
    assert load_catalog(source)["cases"][0]["title"] == first["cases"][0]["title"]
    source.write_text('{"cases": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_catalog(source)


def test_render_escapes_untrusted_markdown_table_content():
    data = load_catalog()
    data["cases"][0]["title"] = "一个 | 分隔符\n<script>bad()</script>"
    actual = render_markdown(data)
    assert "一个 &#124; 分隔符<br>&lt;script&gt;bad()&lt;/script&gt;" in actual
    assert "<script>bad()</script>" not in actual
