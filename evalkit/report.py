"""Turn case results into a summary a person can read in a terminal or a CI log."""
from __future__ import annotations

from typing import Dict, List

from .scorers import CaseResult

SCORERS = ("format", "grounding", "accuracy")


def summarize(results: List[CaseResult]) -> Dict[str, float]:
    n = len(results) or 1
    summary = {name: round(sum(r.scores[name].passed for r in results) / n, 3) for name in SCORERS}
    summary["mean_grounding"] = round(sum(r.scores["grounding"].value for r in results) / n, 3)
    summary["pass_rate"] = round(sum(r.passed for r in results) / n, 3)
    summary["cases"] = len(results)
    return summary


def format_table(runs: Dict[str, Dict[str, float]]) -> str:
    cols = ("cases",) + SCORERS + ("pass_rate",)
    header = f"{'run':<18}" + "".join(f"{c:>11}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, s in runs.items():
        cells = [f"{s['cases']:>11}"] + [f"{s[c]:>11.0%}" for c in cols[1:]]
        lines.append(f"{name:<18}" + "".join(cells))
    return "\n".join(lines)


def failures(results: List[CaseResult]) -> List[str]:
    out = []
    for r in results:
        for s in r.scores.values():
            if not s.passed:
                out.append(f"  {r.case_id:<22} {s.name:<10} {'; '.join(s.reasons)}")
    return out
