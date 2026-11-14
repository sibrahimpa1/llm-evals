"""Offline demo: score two recorded prompt versions against the eval set. No API key needed.

    python -m evalkit.demo
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional

from .cases import DATA_DIR, load_cases, load_kb, load_recorded
from .report import failures, format_table
from .adapters import ReplayAdapter
from .runlog import RunLog
from .runner import format_cost_table, run_eval
from .scorers import CaseResult, evaluate_case
from .guardrails import format_guard_table, guard_failures, run_guardrails, summarize_guard
from .gate import DEFAULT_BASELINE, compare, load_baseline, take_snapshot
from .judge import ReplayJudge, calibrate, format_calibration, judge_runs, load_labels

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
    log = RunLog(":memory:")
    for name in names:
        _, results, summaries[name] = run_eval(ReplayAdapter(name), log, label=name)
        details[name] = failures(results)
    log.close()
    print(format_table(summaries))
    print("\nCost and latency (replay: tokens estimated, latency simulated)")
    print(format_cost_table(summaries))
    for name in names:
        print(f"\nFailures in {name}: {len(details[name])}")
        print("\n".join(details[name]) or "  none")

    print("\nGuardrails (prompt injection, PII, refusals)")
    guard = {name: run_guardrails(ReplayAdapter(name)) for name in names}
    print(format_guard_table({n: summarize_guard(r) for n, r in guard.items()}))
    for name, results in guard.items():
        lines = guard_failures(results)
        print(f"\nGuardrail failures in {name}: {len(lines)}")
        print("\n".join(lines) or "  none")

    print("\nLLM judge calibrated against hand labels (recorded verdicts)")
    scores, _ = judge_runs(ReplayJudge())
    rows, best = calibrate(scores, load_labels())
    print(format_calibration(rows, best))

    print("\nRegression gate against baselines/main.json")
    baseline = load_baseline(DEFAULT_BASELINE)
    for name in names:
        res = compare(baseline, take_snapshot(f"replay:{name}"))
        verdict = "PASS" if res.passed else f"FAIL ({len(res.failures)} regressions)"
        print(f"  replay:{name:<14} {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
