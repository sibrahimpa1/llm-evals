"""Deterministic scorers: format, grounding and accuracy.

The feature under test must reply with JSON: {"answer": str, "citations": [doc_id], "escalate": bool}.
These checks use plain code, not a model, so they are free, fast and give the same result every run.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .cases import EvalCase

REQUIRED = {"answer": str, "citations": list, "escalate": bool}
_FIGURE = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass
class Score:
    name: str
    value: float
    passed: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class CaseResult:
    case_id: str
    scores: Dict[str, Score]

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scores.values())


def parse_output(raw: str) -> Tuple[Optional[dict], List[str]]:
    """Parse the model output and check the contract. Returns (parsed or None, problems)."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, ["output is not valid JSON"]
    if not isinstance(data, dict):
        return None, ["output is JSON but not an object"]
    problems = []
    for key, typ in REQUIRED.items():
        if key not in data:
            problems.append(f"missing key {key!r}")
        elif not isinstance(data[key], typ):
            problems.append(f"{key!r} should be {typ.__name__}")
    if not problems and not all(isinstance(c, str) for c in data["citations"]):
        problems.append("citations must be doc ids (strings)")
    return (data if not problems else None), problems


def score_format(raw: str) -> Score:
    data, problems = parse_output(raw)
    return Score("format", 1.0 if data is not None else 0.0, data is not None, problems)


def _figures(text: str) -> List[str]:
    return _FIGURE.findall(text)


def score_grounding(data: dict, case: EvalCase, kb: Dict[str, str]) -> Score:
    """Citations must point at retrieved docs, and every figure in the answer must appear in a cited doc.

    Invented numbers (prices, days, limits) are the most common and most costly hallucination in a
    support bot, and they are cheap to catch without a model.
    """
    reasons: List[str] = []
    citations = data["citations"]
    bad = [c for c in citations if c not in case.context_docs]
    if bad:
        reasons.append(f"cites docs that were not retrieved: {bad}")
    cited_text = " ".join(kb.get(c, "") for c in citations if c not in bad)
    known = set(_figures(cited_text))
    figures = _figures(data["answer"])
    unsupported = [f for f in figures if f not in known]
    if unsupported:
        reasons.append(f"figures not found in cited docs: {unsupported}")
    if figures and not citations:
        reasons.append("answer states figures but cites nothing")
    total = len(figures) + len(citations)
    good = (len(figures) - len(unsupported)) + (len(citations) - len(bad))
    value = 1.0 if total == 0 else good / total
    return Score("grounding", round(value, 3), not reasons, reasons)


def score_accuracy(data: dict, case: EvalCase) -> Score:
    """Required facts present, forbidden claims absent, and the escalation decision right."""
    answer = data["answer"].lower()
    reasons: List[str] = []
    missing = [p for p in case.must_include if p.lower() not in answer]
    if missing:
        reasons.append(f"missing facts: {missing}")
    forbidden = [p for p in case.must_not_include if p.lower() in answer]
    if forbidden:
        reasons.append(f"says what it must not: {forbidden}")
    if data["escalate"] != case.expect_escalate:
        want = "hand off to a person" if case.expect_escalate else "answer without handing off"
        reasons.append(f"escalation wrong: should {want}")
    found = 1.0 if not case.must_include else (len(case.must_include) - len(missing)) / len(case.must_include)
    value = 0.0 if forbidden or data["escalate"] != case.expect_escalate else found
    return Score("accuracy", round(value, 3), not reasons, reasons)


def evaluate_case(raw: str, case: EvalCase, kb: Dict[str, str]) -> CaseResult:
    fmt = score_format(raw)
    if not fmt.passed:
        skipped = ["not scored: output broke the format"]
        return CaseResult(case.id, {
            "format": fmt,
            "grounding": Score("grounding", 0.0, False, list(skipped)),
            "accuracy": Score("accuracy", 0.0, False, list(skipped)),
        })
    data, _ = parse_output(raw)
    return CaseResult(case.id, {
        "format": fmt,
        "grounding": score_grounding(data, case, kb),
        "accuracy": score_accuracy(data, case),
    })
