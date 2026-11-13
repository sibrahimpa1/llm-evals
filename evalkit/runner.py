"""Run an adapter over the eval set, score every answer and write it to the run log.

    python -m evalkit.runner replay:prompt_v1 replay:prompt_v2 --db runs/evals.sqlite
    python -m evalkit.runner claude openai          # live, needs keys in the environment
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from .adapters import AdapterError, make_adapter
from .cases import load_cases, load_kb
from .pricing import cost_usd, load_prices
from .report import format_table, summarize
from .runlog import RunLog
from .scorers import CaseResult, Score, evaluate_case


def percentile(values: List[float], q: float) -> float:
    """Nearest-rank percentile; small eval sets make interpolation misleading."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(-(-q * len(ordered) // 1)))
    return ordered[min(rank, len(ordered)) - 1]


def run_eval(adapter, log: RunLog, label: Optional[str] = None) -> Tuple[int, List[CaseResult], Dict[str, float]]:
    kb = load_kb()
    cases = load_cases(kb=kb)
    prices = load_prices()
    run_id = log.start_run(label or adapter.model, adapter.model)
    results: List[CaseResult] = []
    costs: List[float] = []
    latencies: List[float] = []
    for case in cases:
        try:
            gen = adapter.generate(case, kb)
        except AdapterError:
            raise
        except Exception as exc:  # one bad call is a failed case, not a crashed run
            err = f"{type(exc).__name__}: {exc}"
            failed = Score("format", 0.0, False, [f"call failed: {err}"])
            result = CaseResult(case.id, {
                "format": failed,
                "grounding": Score("grounding", 0.0, False, ["not scored: call failed"]),
                "accuracy": Score("accuracy", 0.0, False, ["not scored: call failed"])})
            results.append(result)
            log.log_answer(run_id, case.id, "", False, {k: asdict(v) for k, v in result.scores.items()},
                           0, 0, 0.0, 0.0, error=err)
            continue
        result = evaluate_case(gen.output, case, kb)
        cost = cost_usd(gen.model, gen.input_tokens, gen.output_tokens, prices)
        results.append(result)
        costs.append(cost)
        latencies.append(gen.latency_ms)
        log.log_answer(run_id, case.id, gen.output, result.passed,
                       {k: asdict(v) for k, v in result.scores.items()},
                       gen.input_tokens, gen.output_tokens, cost, gen.latency_ms, gen.simulated)
    log.finish_run(run_id)
    summary = summarize(results)
    summary.update(cost_usd=round(sum(costs), 6),
                   p50_ms=percentile(latencies, 0.5),
                   p95_ms=percentile(latencies, 0.95))
    return run_id, results, summary


def format_cost_table(runs: Dict[str, Dict[str, float]]) -> str:
    header = f"{'run':<18}{'pass_rate':>11}{'cost_usd':>12}{'p50_ms':>10}{'p95_ms':>10}"
    lines = [header, "-" * len(header)]
    for name, s in runs.items():
        lines.append(f"{name:<18}{s['pass_rate']:>11.0%}{s['cost_usd']:>12.4f}{s['p50_ms']:>10.0f}{s['p95_ms']:>10.0f}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval set against one or more models.")
    parser.add_argument("adapters", nargs="*", default=["replay:prompt_v1", "replay:prompt_v2"])
    parser.add_argument("--db", default="runs/evals.sqlite", help="SQLite run log (':memory:' for none)")
    args = parser.parse_args(argv)
    log = RunLog(args.db)
    summaries: Dict[str, Dict[str, float]] = {}
    try:
        for spec in args.adapters:
            try:
                adapter = make_adapter(spec)
            except AdapterError as exc:
                print(f"skipped {spec}: {exc}", file=sys.stderr)
                continue
            _, _, summary = run_eval(adapter, log, label=spec)
            summaries[spec] = summary
    finally:
        log.close()
    if not summaries:
        return 1
    print(format_table(summaries))
    print()
    print(format_cost_table(summaries))
    if args.db != ":memory:":
        print(f"\nRun log: {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
