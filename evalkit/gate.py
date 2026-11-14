"""Regression gate: compare a run against the committed baseline and fail the pull request on a regression.

    python -m evalkit.gate replay:prompt_v1                      # check a run against baselines/main.json
    python -m evalkit.gate replay:prompt_v1 --write-baseline     # accept a run as the new baseline
    python -m evalkit.gate claude --summary report.md            # live model, markdown report to a file

The baseline is a small JSON file in git, so a change to it shows up in review like any other diff:
accepting worse quality has to be a visible decision, not a silent one.

What fails the gate:
- a case that passed in the baseline and fails now (eval set or guardrail suite)
- a guardrail case regressing, with no tolerance at all
- the eval pass rate dropping more than the policy allows
- cost or p95 latency growing more than the policy allows
- a baseline case missing from the run (deleting a failing case is not a fix)

Exit codes: 0 pass, 1 regression, 2 setup problem (no baseline, unknown adapter).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .adapters import AdapterError, make_adapter
from .guardrails import run_guardrails, summarize_guard
from .runlog import RunLog
from .runner import run_eval

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = ROOT / "baselines" / "main.json"
DEFAULT_POLICY = {
    "max_pass_rate_drop": 0.0,   # eval pass rate may not fall at all
    "max_cost_increase": 0.25,   # +25% total cost
    "max_p95_increase": 0.30,    # +30% p95 latency
}
BASELINE_FORMAT = 1


@dataclass
class Snapshot:
    """Everything the gate compares, for one adapter."""
    adapter: str
    model: str
    metrics: Dict[str, float]
    eval_cases: Dict[str, bool]
    guard_cases: Dict[str, bool]

    def to_json(self, policy: Dict[str, float]) -> dict:
        return {"format": BASELINE_FORMAT, "adapter": self.adapter, "model": self.model,
                "policy": policy, "metrics": self.metrics,
                "eval_cases": dict(sorted(self.eval_cases.items())),
                "guard_cases": dict(sorted(self.guard_cases.items()))}


@dataclass
class GateResult:
    failures: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    rows: List[tuple] = field(default_factory=list)  # (metric, baseline, current, delta text)

    @property
    def passed(self) -> bool:
        return not self.failures


def take_snapshot(spec: str) -> Snapshot:
    adapter = make_adapter(spec)
    log = RunLog(":memory:")
    try:
        _, results, summary = run_eval(adapter, log, label=spec)
    finally:
        log.close()
    guard = run_guardrails(adapter)
    gsum = summarize_guard(guard)
    metrics = {
        "eval_pass_rate": summary["pass_rate"],
        "format": summary["format"],
        "grounding": summary["grounding"],
        "accuracy": summary["accuracy"],
        "guard_pass_rate": gsum["pass_rate"],
        "cost_usd": summary["cost_usd"],
        "p95_ms": summary["p95_ms"],
    }
    return Snapshot(spec, adapter.model, metrics,
                    {r.case_id: r.passed for r in results},
                    {r.case_id: r.passed for r in guard})


def load_baseline(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != BASELINE_FORMAT:
        raise ValueError(f"{path}: baseline format {data.get('format')!r}, expected {BASELINE_FORMAT}")
    for key in ("metrics", "eval_cases", "guard_cases"):
        if not isinstance(data.get(key), dict):
            raise ValueError(f"{path}: baseline has no {key!r}")
    return data


def _relative_growth(old: float, new: float) -> float:
    if old <= 0:
        return 0.0 if new <= 0 else float("inf")
    return (new - old) / old


def _compare_cases(label: str, old: Dict[str, bool], new: Dict[str, bool], allow: List[str],
                   res: GateResult) -> None:
    for case_id, was in sorted(old.items()):
        if case_id not in new:
            res.failures.append(f"{label} case `{case_id}` is in the baseline but missing from the run")
            continue
        now = new[case_id]
        if was and not now:
            if case_id in allow:
                res.notes.append(f"{label} case `{case_id}` regressed (allowed with --allow)")
            else:
                res.failures.append(f"{label} case `{case_id}` passed in the baseline and fails now")
        elif not was and now:
            res.improvements.append(f"{label} case `{case_id}` now passes")
    for case_id in sorted(set(new) - set(old)):
        state = "passes" if new[case_id] else "fails"
        res.notes.append(f"new {label} case `{case_id}` {state} (not in the baseline yet)")


def compare(baseline: dict, current: Snapshot, allow: Optional[List[str]] = None) -> GateResult:
    allow = list(allow or [])
    policy = {**DEFAULT_POLICY, **baseline.get("policy", {})}
    old, new = baseline["metrics"], current.metrics
    res = GateResult()

    _compare_cases("eval", baseline["eval_cases"], current.eval_cases, allow, res)
    # Guardrail cases get no --allow escape hatch: a new prompt-injection or PII leak never ships quietly.
    _compare_cases("guardrail", baseline["guard_cases"], current.guard_cases, [], res)

    for name in ("eval_pass_rate", "format", "grounding", "accuracy", "guard_pass_rate"):
        o, n = float(old.get(name, 0.0)), float(new.get(name, 0.0))
        res.rows.append((name, f"{o:.0%}", f"{n:.0%}", f"{(n - o) * 100:+.0f} pts"))
    drop = float(old["eval_pass_rate"]) - float(new["eval_pass_rate"])
    if drop > policy["max_pass_rate_drop"] + 1e-9:
        res.failures.append(f"eval pass rate fell {drop * 100:.0f} pts "
                            f"(policy allows {policy['max_pass_rate_drop'] * 100:.0f})")
    if float(new["guard_pass_rate"]) < float(old["guard_pass_rate"]):
        res.failures.append("guardrail pass rate fell")

    for name, key, fmt in (("cost_usd", "max_cost_increase", "{:.4f}"), ("p95_ms", "max_p95_increase", "{:.0f}")):
        o, n = float(old[name]), float(new[name])
        growth = _relative_growth(o, n)
        delta = "n/a" if growth == float("inf") else f"{growth:+.0%}"
        res.rows.append((name, fmt.format(o), fmt.format(n), delta))
        if growth > policy[key] + 1e-9:
            res.failures.append(f"{name} grew {delta} (policy allows +{policy[key]:.0%})")
        elif growth < -0.05:
            res.improvements.append(f"{name} down {delta}")
    return res


def format_markdown(res: GateResult, baseline: dict, current: Snapshot) -> str:
    verdict = "PASS" if res.passed else "FAIL"
    out = [f"## Eval gate: {verdict}", "",
           f"Run `{current.adapter}` against baseline `{baseline.get('adapter', '?')}`.", "",
           "| metric | baseline | this run | change |", "|---|---:|---:|---:|"]
    out += [f"| {m} | {b} | {c} | {d} |" for m, b, c, d in res.rows]
    for title, items in (("Regressions", res.failures), ("Improvements", res.improvements), ("Notes", res.notes)):
        if items:
            out += ["", f"### {title}", ""] + [f"- {i}" for i in items]
    if not res.passed:
        out += ["", "If this change is meant to trade quality for cost, update `baselines/main.json` "
                "with `--write-baseline` in the same pull request so the trade-off is reviewed."]
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fail on a regression against the committed eval baseline.")
    parser.add_argument("adapter", nargs="?", default="replay:prompt_v1")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write-baseline", action="store_true", help="save this run as the new baseline")
    parser.add_argument("--allow", action="append", default=[], metavar="CASE_ID",
                        help="accept a known eval-case regression (not available for guardrail cases)")
    parser.add_argument("--summary", type=Path, help="also write the markdown report here")
    args = parser.parse_args(argv)

    try:
        current = take_snapshot(args.adapter)
    except AdapterError as exc:
        print(f"gate: {exc}", file=sys.stderr)
        return 2

    if args.write_baseline:
        policy = DEFAULT_POLICY
        if args.baseline.exists():
            policy = {**DEFAULT_POLICY, **load_baseline(args.baseline).get("policy", {})}
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(current.to_json(policy), indent=2) + "\n", encoding="utf-8")
        print(f"Baseline written: {args.baseline} ({args.adapter})")
        return 0

    try:
        baseline = load_baseline(args.baseline)
    except (OSError, ValueError) as exc:
        print(f"gate: cannot read baseline: {exc}", file=sys.stderr)
        return 2

    res = compare(baseline, current, allow=args.allow)
    report = format_markdown(res, baseline, current)
    print(report)
    targets = [args.summary] if args.summary else []
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        targets.append(Path(step_summary))
    for target in targets:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(report)
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
