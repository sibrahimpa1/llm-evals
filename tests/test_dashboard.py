import json

import pytest

from evalkit.dashboard import CaseRow, ModelReport, collect, main, render
from evalkit.gate import DEFAULT_BASELINE, load_baseline


@pytest.fixture(scope="module")
def reports():
    baseline = load_baseline(DEFAULT_BASELINE)
    return [collect("replay:prompt_v1", baseline), collect("replay:prompt_v2", baseline)]


def test_collect_keeps_per_case_detail(reports):
    v1, v2 = reports
    assert v1.summary["pass_rate"] == 1.0 and v1.gate == "PASS"
    assert v2.gate == "FAIL" and v2.gate_failures
    assert len(v1.cases) == v1.summary["cases"]
    assert all(c.cost_usd > 0 and c.latency_ms > 0 for c in v1.cases)
    assert v1.simulated
    failed = [c for c in v2.cases if not c.passed]
    assert failed and all(c.reasons for c in failed)


def test_render_shows_every_model_and_case(reports):
    page = render(reports, generated_at="2026-01-01 00:00 UTC")
    assert page.startswith("<!doctype html>")
    for rep in reports:
        assert rep.spec in page
        for case in rep.cases + rep.guard_cases:
            assert case.case_id in page
    assert "gate PASS" in page and "gate FAIL" in page
    assert "<script" not in page  # static: no JavaScript, opens offline


def test_model_output_is_escaped():
    evil = '<script>alert("x")</script>'
    rep = ModelReport(spec="replay:<b>evil</b>", model="m",
                      summary={"format": 1, "grounding": 1, "accuracy": 0, "pass_rate": 0, "cases": 1,
                               "cost_usd": 0.0, "p50_ms": 0, "p95_ms": 0, "mean_grounding": 1},
                      guard={"format": 1, "prompt_leak": 1, "injection": 1, "pii": 1, "refusal": 1,
                             "pass_rate": 1, "cases": 1},
                      cases=[CaseRow("c1", False, [f"accuracy: said {evil}"])],
                      guard_cases=[CaseRow("g1", True, [])], simulated=False)
    page = render([rep])
    assert evil not in page and "<b>evil</b>" not in page
    assert "&lt;script&gt;" in page


def test_cli_writes_html_and_json(tmp_path, capsys):
    out, js = tmp_path / "d.html", tmp_path / "d.json"
    assert main(["replay:prompt_v1", "replay:prompt_v2", "--out", str(out), "--json", str(js)]) == 0
    assert out.read_text(encoding="utf-8").count('class="card"') == 2
    data = json.loads(js.read_text(encoding="utf-8"))
    assert [d["spec"] for d in data] == ["replay:prompt_v1", "replay:prompt_v2"]
    assert "gate PASS" in capsys.readouterr().out


def test_cli_skips_unknown_adapter(tmp_path):
    assert main(["nope", "--out", str(tmp_path / "x.html")]) == 1
