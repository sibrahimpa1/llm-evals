"""Static dashboard: quality, guardrails, cost and latency per model in one HTML file.

    python -m evalkit.dashboard                                   # offline, the two recorded prompts
    python -m evalkit.dashboard replay:prompt_v1 claude openai    # live adapters are added when keys are set
    python -m evalkit.dashboard --out runs/dashboard.html

The page is a single file with inline CSS and SVG and no JavaScript libraries, so it opens offline,
can be attached to a CI run as an artifact and never needs a server. Every model output is escaped
before it is written, because red-team answers are untrusted text.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .adapters import AdapterError, make_adapter
from .gate import DEFAULT_BASELINE, Snapshot, compare, load_baseline
from .guardrails import CHECKS, run_guardrails, summarize_guard
from .runlog import RunLog
from .runner import run_eval
from .report import SCORERS

DEFAULT_OUT = Path("runs") / "dashboard.html"
PALETTE = ("#3b6fd4", "#d9822b", "#2f9e72", "#a64ac9", "#c94a4a", "#6b7a8f")


@dataclass
class CaseRow:
    case_id: str
    passed: bool
    reasons: List[str]
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass
class ModelReport:
    spec: str
    model: str
    summary: Dict[str, float]
    guard: Dict[str, float]
    cases: List[CaseRow]
    guard_cases: List[CaseRow]
    simulated: bool
    gate: Optional[str] = None            # "PASS", "FAIL", or None without a baseline
    gate_failures: List[str] = field(default_factory=list)


def collect(spec: str, baseline: Optional[dict] = None) -> ModelReport:
    """Run one adapter over the eval set and the guardrail suite and keep per-case detail."""
    adapter = make_adapter(spec)
    log = RunLog(":memory:")
    try:
        run_id, results, summary = run_eval(adapter, log, label=spec)
        rows = {r["case_id"]: r for r in log.answers(run_id)}
    finally:
        log.close()
    cases = []
    for r in results:
        reasons = [f"{s.name}: {'; '.join(s.reasons)}" for s in r.scores.values() if not s.passed]
        row = rows.get(r.case_id)
        cases.append(CaseRow(r.case_id, r.passed, reasons,
                             float(row["cost_usd"]) if row else 0.0,
                             float(row["latency_ms"]) if row else 0.0))
    simulated = any(bool(r["simulated"]) for r in rows.values())

    guard_results = run_guardrails(adapter)
    gsum = summarize_guard(guard_results)
    guard_cases = [CaseRow(g.case_id, g.passed,
                           [f"{s.name}: {'; '.join(s.reasons)}" for s in g.checks.values() if not s.passed])
                   for g in guard_results]

    report = ModelReport(spec, adapter.model, summary, gsum, cases, guard_cases, simulated)
    if baseline is not None:
        snap = Snapshot(spec, adapter.model, {
            "eval_pass_rate": summary["pass_rate"], "format": summary["format"],
            "grounding": summary["grounding"], "accuracy": summary["accuracy"],
            "guard_pass_rate": gsum["pass_rate"], "cost_usd": summary["cost_usd"],
            "p95_ms": summary["p95_ms"]},
            {c.case_id: c.passed for c in cases}, {g.case_id: g.passed for g in guard_cases})
        res = compare(baseline, snap)
        report.gate = "PASS" if res.passed else "FAIL"
        report.gate_failures = res.failures
    return report


# ---------------------------------------------------------------- rendering

def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _pct(v: float) -> str:
    return f"{v:.0%}"


def _bars(reports: List[ModelReport], metrics: List[tuple]) -> str:
    """Grouped horizontal bars, one group per metric, one bar per model. Values are 0..1."""
    bar_h, gap, label_w, width = 14, 4, 120, 560
    plot_w = width - label_w - 50
    group_h = len(reports) * (bar_h + gap) + 14
    height = group_h * len(metrics)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
             f'aria-label="pass rate per metric and model">']
    for gi, (label, getter) in enumerate(metrics):
        y0 = gi * group_h
        parts.append(f'<text x="0" y="{y0 + 14}" class="axis">{_e(label)}</text>')
        for mi, rep in enumerate(reports):
            v = max(0.0, min(1.0, float(getter(rep))))
            y = y0 + mi * (bar_h + gap)
            w = round(plot_w * v, 1)
            color = PALETTE[mi % len(PALETTE)]
            parts.append(f'<rect x="{label_w}" y="{y}" width="{plot_w}" height="{bar_h}" class="track"/>')
            parts.append(f'<rect x="{label_w}" y="{y}" width="{w}" height="{bar_h}" fill="{color}" rx="2">'
                         f'<title>{_e(rep.spec)}: {_pct(v)}</title></rect>')
            parts.append(f'<text x="{label_w + plot_w + 6}" y="{y + bar_h - 3}" class="val">{_pct(v)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _scatter(reports: List[ModelReport]) -> str:
    """Cost (x) against eval pass rate (y): the trade-off a model choice is really about."""
    width, height, pad = 560, 260, 46
    max_cost = max((r.summary["cost_usd"] for r in reports), default=0.0) or 1.0
    max_cost *= 1.15
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" class="chart scatter" '
             f'aria-label="cost against pass rate">',
             f'<line x1="{pad}" y1="{height - pad}" x2="{width - 10}" y2="{height - pad}" class="grid"/>',
             f'<line x1="{pad}" y1="10" x2="{pad}" y2="{height - pad}" class="grid"/>',
             f'<text x="{width - 10}" y="{height - 12}" text-anchor="end" class="axis">total cost, USD</text>',
             f'<text x="{pad - 8}" y="16" text-anchor="end" class="axis">100%</text>',
             f'<text x="{pad - 8}" y="{height - pad}" text-anchor="end" class="axis">0%</text>',
             f'<text x="{pad}" y="{height - pad + 16}" class="axis">0</text>',
             f'<text x="{width - 10}" y="{height - pad + 16}" text-anchor="end" class="axis">{max_cost:.4f}</text>']
    for i, rep in enumerate(reports):
        x = pad + (width - 10 - pad) * rep.summary["cost_usd"] / max_cost
        y = (height - pad) - (height - pad - 10) * rep.summary["pass_rate"]
        color = PALETTE[i % len(PALETTE)]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}">'
                     f'<title>{_e(rep.spec)}: {_pct(rep.summary["pass_rate"])}, '
                     f'${rep.summary["cost_usd"]:.4f}</title></circle>')
        # labels right of the middle sit left of their dot so they never run off the chart
        if x > width / 2:
            parts.append(f'<text x="{x - 11:.1f}" y="{y + 4:.1f}" text-anchor="end" class="val">'
                         f'{_e(rep.spec)}</text>')
        else:
            parts.append(f'<text x="{x + 11:.1f}" y="{y + 4:.1f}" class="val">{_e(rep.spec)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _matrix(reports: List[ModelReport], attr: str, title: str) -> str:
    ids: List[str] = []
    for rep in reports:
        for c in getattr(rep, attr):
            if c.case_id not in ids:
                ids.append(c.case_id)
    head = "".join(f"<th>{_e(r.spec)}</th>" for r in reports)
    body = []
    for cid in ids:
        cells = []
        for rep in reports:
            row = next((c for c in getattr(rep, attr) if c.case_id == cid), None)
            if row is None:
                cells.append('<td class="na">n/a</td>')
            elif row.passed:
                cells.append('<td class="ok" title="passed">pass</td>')
            else:
                why = " | ".join(row.reasons)
                cells.append(f'<td class="bad" title="{_e(why)}">fail<span>{_e(why)}</span></td>')
        body.append(f"<tr><th scope=\"row\"><code>{_e(cid)}</code></th>{''.join(cells)}</tr>")
    return (f"<h2>{_e(title)}</h2><div class=\"scroll\"><table class=\"matrix\"><thead><tr><th>case</th>"
            f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d2330;--muted:#5d6675;--line:#e2e4e8;--ok:#1f7a55;--okbg:#e5f4ec;
--bad:#a8322f;--badbg:#fbe9e7;--track:#eceef1}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1c2027;--ink:#e6e8eb;--muted:#9aa3b0;--line:#2c323b;
--ok:#5fd19e;--okbg:#173327;--bad:#ff8a80;--badbg:#3a1d1c;--track:#2a2f37}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 48px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:17px;margin:32px 0 10px}p.sub{color:var(--muted);margin:0 0 20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card h3{margin:0 0 8px;font-size:15px;display:flex;align-items:center;gap:8px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.kv{display:grid;grid-template-columns:1fr auto;gap:2px 12px;font-size:14px}.kv span{color:var(--muted)}
.badge{font-size:12px;font-weight:600;padding:1px 8px;border-radius:99px;margin-left:auto}
.badge.PASS{background:var(--okbg);color:var(--ok)}.badge.FAIL{background:var(--badbg);color:var(--bad)}
.why{margin:8px 0 0;padding-left:18px;font-size:13px;color:var(--bad)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.chart{width:100%;height:auto;display:block}.chart.scatter{max-width:640px;margin:0 auto}
.two>div,.cards>div{min-width:0}.chart .axis{font-size:12px;fill:var(--muted)}.chart .val{font-size:11px;fill:var(--ink)}
.chart .track{fill:var(--track)}.chart .grid{stroke:var(--line)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
td.ok{color:var(--ok);background:var(--okbg)}td.bad{color:var(--bad);background:var(--badbg)}
td.bad span{display:block;color:var(--ink);font-size:12px;max-width:340px}td.na{color:var(--muted)}
.note{color:var(--muted);font-size:13px}
"""


def render(reports: List[ModelReport], generated_at: Optional[str] = None) -> str:
    generated_at = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cards = []
    for i, rep in enumerate(reports):
        s, g = rep.summary, rep.guard
        badge = f'<span class="badge {rep.gate}">gate {rep.gate}</span>' if rep.gate else ""
        why = ""
        if rep.gate_failures:
            items = "".join(f"<li>{_e(f)}</li>" for f in rep.gate_failures[:4])
            more = len(rep.gate_failures) - 4
            if more > 0:
                items += f"<li>and {more} more</li>"
            why = f'<ul class="why">{items}</ul>'
        cards.append(
            f'<div class="card"><h3><span class="dot" style="background:{PALETTE[i % len(PALETTE)]}"></span>'
            f'{_e(rep.spec)}{badge}</h3><div class="kv">'
            f'<span>eval pass rate</span><b>{_pct(s["pass_rate"])}</b>'
            f'<span>guardrail pass rate</span><b>{_pct(g["pass_rate"])}</b>'
            f'<span>total cost</span><b>${s["cost_usd"]:.4f}</b>'
            f'<span>latency p50 / p95</span><b>{s["p50_ms"]:.0f} / {s["p95_ms"]:.0f} ms</b>'
            f'<span>cases</span><b>{s["cases"]} + {g["cases"]} red-team</b></div>{why}</div>')
    quality = [(name, (lambda r, n=name: r.summary[n])) for name in SCORERS]
    quality.append(("all checks", lambda r: r.summary["pass_rate"]))
    guard = [(name, (lambda r, n=name: r.guard[n])) for name in CHECKS if name != "format"]
    guard.append(("all checks", lambda r: r.guard["pass_rate"]))
    simulated = any(r.simulated for r in reports)
    note = ('<p class="note">Replay runs use estimated tokens and simulated latency; prices in '
            '<code>data/pricing.json</code> are illustrative.</p>') if simulated else ""
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>Eval Dashboard</title><style>{CSS}</style></head><body><main>"
        "<h1>Eval dashboard</h1>"
        f"<p class=\"sub\">Quality, guardrails, cost and latency per model. Generated {_e(generated_at)}.</p>"
        f"<div class=\"cards\">{''.join(cards)}</div>"
        "<div class=\"two\">"
        f"<div><h2>Quality checks</h2><div class=\"panel\">{_bars(reports, quality)}</div></div>"
        f"<div><h2>Guardrails</h2><div class=\"panel\">{_bars(reports, guard)}</div></div>"
        "</div>"
        f"<h2>Cost against quality</h2><div class=\"panel\">{_scatter(reports)}</div>{note}"
        f"{_matrix(reports, 'cases', 'Eval cases')}"
        f"{_matrix(reports, 'guard_cases', 'Red-team cases')}"
        "</main></body></html>\n")


def to_json(reports: List[ModelReport]) -> str:
    return json.dumps([{
        "spec": r.spec, "model": r.model, "summary": r.summary, "guard": r.guard, "gate": r.gate,
        "gate_failures": r.gate_failures, "simulated": r.simulated,
        "cases": [c.__dict__ for c in r.cases], "guard_cases": [c.__dict__ for c in r.guard_cases],
    } for r in reports], indent=2)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Write a static HTML dashboard comparing models.")
    parser.add_argument("adapters", nargs="*", default=["replay:prompt_v1", "replay:prompt_v2"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", type=Path, help="also write the numbers as JSON")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                        help="show each model's gate verdict against this baseline")
    args = parser.parse_args(argv)

    baseline = None
    if args.baseline and args.baseline.exists():
        baseline = load_baseline(args.baseline)
    reports = []
    for spec in args.adapters:
        try:
            reports.append(collect(spec, baseline))
        except AdapterError as exc:
            print(f"skipped {spec}: {exc}", file=sys.stderr)
    if not reports:
        print("dashboard: no adapter could run", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(reports), encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(to_json(reports) + "\n", encoding="utf-8")
    for r in reports:
        gate = f"  gate {r.gate}" if r.gate else ""
        print(f"{r.spec:<20} eval {_pct(r.summary['pass_rate']):>5}  guard {_pct(r.guard['pass_rate']):>5}  "
              f"${r.summary['cost_usd']:.4f}  p95 {r.summary['p95_ms']:.0f} ms{gate}")
    print(f"\nDashboard: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
