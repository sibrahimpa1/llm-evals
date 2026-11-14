import json

import pytest

from evalkit.gate import DEFAULT_BASELINE, Snapshot, compare, format_markdown, load_baseline, main, take_snapshot


def snap(**over):
    base = dict(adapter="x", model="x",
                metrics={"eval_pass_rate": 1.0, "format": 1.0, "grounding": 1.0, "accuracy": 1.0,
                         "guard_pass_rate": 1.0, "cost_usd": 0.01, "p95_ms": 1000.0},
                eval_cases={"a": True, "b": True}, guard_cases={"g1": True})
    base.update(over)
    return Snapshot(**base)


def baseline_of(s: Snapshot, **policy):
    return s.to_json({"max_pass_rate_drop": 0.0, "max_cost_increase": 0.25, "max_p95_increase": 0.30, **policy})


def test_identical_run_passes():
    s = snap()
    assert compare(baseline_of(s), s).passed


def test_case_regression_fails_and_can_be_allowed():
    base = baseline_of(snap())
    cur = snap(eval_cases={"a": True, "b": False},
               metrics={**snap().metrics, "eval_pass_rate": 1.0})
    res = compare(base, cur)
    assert not res.passed
    assert any("`b` passed in the baseline" in f for f in res.failures)
    assert compare(base, cur, allow=["b"]).passed


def test_guardrail_regression_cannot_be_allowed():
    base = baseline_of(snap())
    cur = snap(guard_cases={"g1": False})
    res = compare(base, cur, allow=["g1"])
    assert not res.passed
    assert any("guardrail case `g1`" in f for f in res.failures)


def test_missing_case_fails_new_case_is_a_note():
    base = baseline_of(snap())
    cur = snap(eval_cases={"a": True, "c": False})
    res = compare(base, cur)
    assert any("`b` is in the baseline but missing" in f for f in res.failures)
    assert any("new eval case `c` fails" in n for n in res.notes)


def test_pass_rate_drop_respects_policy():
    cur = snap(metrics={**snap().metrics, "eval_pass_rate": 0.95})
    assert not compare(baseline_of(snap()), cur).passed
    assert compare(baseline_of(snap(), max_pass_rate_drop=0.10), cur).passed


def test_cost_and_latency_growth():
    cur = snap(metrics={**snap().metrics, "cost_usd": 0.013, "p95_ms": 1200.0})
    res = compare(baseline_of(snap()), cur)
    assert any(f.startswith("cost_usd grew") for f in res.failures)
    assert not any(f.startswith("p95_ms") for f in res.failures)
    cheaper = snap(metrics={**snap().metrics, "cost_usd": 0.005})
    res = compare(baseline_of(snap()), cheaper)
    assert res.passed and any("cost_usd down" in i for i in res.improvements)


def test_improvement_is_reported():
    base = baseline_of(snap(eval_cases={"a": True, "b": False}))
    res = compare(base, snap())
    assert res.passed and any("`b` now passes" in i for i in res.improvements)


def test_markdown_report():
    base = baseline_of(snap())
    cur = snap(guard_cases={"g1": False})
    md = format_markdown(compare(base, cur), base, cur)
    assert md.startswith("## Eval gate: FAIL")
    assert "| eval_pass_rate |" in md and "--write-baseline" in md


def test_committed_baseline_matches_the_good_prompt():
    baseline = load_baseline(DEFAULT_BASELINE)
    assert baseline["adapter"] == "replay:prompt_v1"
    assert compare(baseline, take_snapshot("replay:prompt_v1")).passed


def test_cli_blocks_the_cheaper_prompt(tmp_path, capsys, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "step.md"))
    assert main(["replay:prompt_v2", "--summary", str(summary)]) == 1
    assert "Eval gate: FAIL" in summary.read_text()
    assert "rt-pii-other-customer" in (tmp_path / "step.md").read_text()
    assert main(["replay:prompt_v1"]) == 0


def test_cli_write_baseline_and_setup_errors(tmp_path):
    path = tmp_path / "b.json"
    assert main(["replay:prompt_v2", "--baseline", str(path), "--write-baseline"]) == 0
    assert json.loads(path.read_text())["adapter"] == "replay:prompt_v2"
    assert main(["replay:prompt_v2", "--baseline", str(path)]) == 0
    assert main(["replay:prompt_v1", "--baseline", str(tmp_path / "none.json")]) == 2
    assert main(["replay:no_such_run", "--baseline", str(path)]) == 2


def test_bad_baseline_format(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"format": 99}))
    with pytest.raises(ValueError):
        load_baseline(path)
