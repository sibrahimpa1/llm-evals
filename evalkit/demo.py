"""Offline demo: score two recorded prompt versions against the eval set. No API key needed.

    python -m evalkit.demo
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional

from .cases import DATA_DIR, load_cases, load_kb, load_recorded
from .report import failures, format_table, summarize
from .scorers import CaseResult, evaluate_case

RUNS = ("prompt_v1", "prompt_v2")


def run_recorded(name: str) -> List[CaseResult]:
    kb = load_kb()
    cases = load_cases(kb=kb)
    outputs = load_recorded(DATA_DIR / "recorded" / f"{name}.jsonl")
    missing = [c.id for c in cases if c.id not in outputs]
    if missing:
        raise SystemExit(f"{name}: no recorded output for {missing}")
    return [evaluate_case(outputs[c.id], c, kb) for c in cases]


def main(argv: Optional[List[str]] = None) -> int:
    names = (argv if argv else list(RUNS))
    summaries: Dict[str, Dict[str, float]] = {}
    details: Dict[str, List[str]] = {}
    for name in names:
        results = run_recorded(name)
        summaries[name] = summarize(results)
        details[name] = failures(results)
    print(format_table(summaries))
    for name in names:
        print(f"\nFailures in {name}: {len(details[name])}")
        print("\n".join(details[name]) or "  none")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
