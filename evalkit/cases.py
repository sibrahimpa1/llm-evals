"""The eval set: help-center documents and the cases that test answers built from them."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DOC_ID = re.compile(r"\(doc:\s*([a-z0-9_-]+)\)")


@dataclass(frozen=True)
class EvalCase:
    """One question the AI feature must answer, and what a correct answer looks like."""

    id: str
    question: str
    context_docs: List[str]
    must_include: List[str] = field(default_factory=list)
    must_not_include: List[str] = field(default_factory=list)
    expect_escalate: bool = False
    tags: List[str] = field(default_factory=list)


class EvalSetError(ValueError):
    """The eval set itself is broken (bad JSON, duplicate id, unknown document)."""


def load_kb(kb_dir: Path = DATA_DIR / "kb") -> Dict[str, str]:
    """Read every markdown doc. The id comes from '(doc: id)' in the first line, else the file name."""
    kb: Dict[str, str] = {}
    for path in sorted(Path(kb_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = _DOC_ID.search(text.splitlines()[0] if text else "")
        doc_id = match.group(1) if match else path.stem
        if doc_id in kb:
            raise EvalSetError(f"duplicate doc id {doc_id!r} in {path.name}")
        kb[doc_id] = text
    return kb


def load_cases(path: Path = DATA_DIR / "cases.jsonl", kb: Dict[str, str] = None) -> List[EvalCase]:
    """Load and validate the cases. A broken eval set fails loudly instead of scoring quietly."""
    kb = load_kb() if kb is None else kb
    cases: List[EvalCase] = []
    seen = set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            case = EvalCase(**raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise EvalSetError(f"line {n}: {exc}") from exc
        if case.id in seen:
            raise EvalSetError(f"line {n}: duplicate case id {case.id!r}")
        missing = [d for d in case.context_docs if d not in kb]
        if missing:
            raise EvalSetError(f"line {n}: case {case.id!r} names unknown docs {missing}")
        seen.add(case.id)
        cases.append(case)
    return cases


def load_recorded(path: Path) -> Dict[str, str]:
    """Recorded model outputs, keyed by case id, so the eval can run offline."""
    out: Dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["case_id"]] = row["output"]
    return out
