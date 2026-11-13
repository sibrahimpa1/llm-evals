import json

import pytest

from evalkit.cases import EvalSetError, load_cases, load_kb


def test_kb_ids_come_from_doc_header():
    kb = load_kb()
    assert set(kb) == {"refunds", "billing", "security", "integrations"}


def test_eval_set_loads_and_covers_escalation():
    cases = load_cases()
    assert len(cases) >= 10
    assert any(c.expect_escalate for c in cases)
    assert any(not c.expect_escalate for c in cases)


def _write(tmp_path, rows):
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_duplicate_case_id_is_rejected(tmp_path):
    row = {"id": "a", "question": "q", "context_docs": ["refunds"]}
    with pytest.raises(EvalSetError, match="duplicate"):
        load_cases(_write(tmp_path, [row, row]), kb=load_kb())


def test_unknown_doc_is_rejected(tmp_path):
    row = {"id": "a", "question": "q", "context_docs": ["nope"]}
    with pytest.raises(EvalSetError, match="unknown docs"):
        load_cases(_write(tmp_path, [row]), kb=load_kb())


def test_unknown_field_is_rejected(tmp_path):
    row = {"id": "a", "question": "q", "context_docs": [], "answer_key": "x"}
    with pytest.raises(EvalSetError):
        load_cases(_write(tmp_path, [row]), kb=load_kb())
