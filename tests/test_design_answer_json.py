import pytest
from benchmarks.agent_design.answer_json import extract_object


@pytest.mark.parametrize("answer", [
    '{"T1":"open"}', '```json\n{"T1":"open"}\n```',
    '查询完成。\n```json\n{"T1":"open"}\n```\nT1 的状态为 open。',
    '查询完成。\n{"T1":"open"}\nT1 的状态为 open。',
])
def test_complete_unambiguous_declared_object(answer):
    assert extract_object(answer) == {"T1": "open"}


@pytest.mark.parametrize("answer", [
    '```json\n{"T1":"open"\n```',
    '```json\n{"T1":"open"}\n```\n```json\n{"T1":"closed"}\n```',
    '```python\n{"T1":"open"}\n```',
    'T1 is open', '[{"T1":"open"}]',
    'result {"T1":"open"} or {"T1":"closed"}',
    'result {"outer":{"T1":"open"}',
    '{"T1":"open", "T1":"closed"}', '{"T1":NaN}',
])
def test_does_not_repair_truncation_or_choose_ambiguous_answers(answer):
    assert extract_object(answer) == {}
