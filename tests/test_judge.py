import pytest

from evalkit.adapters import AdapterError
from evalkit.cases import load_cases, load_kb
from evalkit.judge import (ClaudeJudge, JudgeError, ReplayJudge, calibrate, cohen_kappa, disagreements,
                           judge_runs, load_labels, main, parse_verdict)


def test_parse_verdict_handles_fences_and_bad_scores():
    assert parse_verdict('```json\n{"score": 4, "reason": "ok"}\n```').score == 4
    with pytest.raises(JudgeError):
        parse_verdict('{"score": 9}')
    with pytest.raises(JudgeError):
        parse_verdict("looks fine to me")


def test_kappa_edges():
    assert cohen_kappa(5, 0, 0, 5) == 1.0
    assert cohen_kappa(0, 0, 0, 0) == 0.0
    assert cohen_kappa(10, 0, 0, 0) == 0.0  # everyone agrees by chance only


def test_calibration_picks_best_kappa_and_prefers_stricter_on_tie():
    scores = {("r", "a"): 5, ("r", "b"): 4, ("r", "c"): 2, ("r", "d"): 1}
    labels = {("r", "a"): True, ("r", "b"): True, ("r", "c"): False, ("r", "d"): False}
    rows, best = calibrate(scores, labels)
    assert best == 4  # 3 and 4 both separate perfectly; 4 is stricter
    assert {r.threshold: r.kappa for r in rows}[4] == 1.0


def test_recorded_judge_agrees_with_people_but_misses_an_invented_number():
    scores, reasons = judge_runs(ReplayJudge())
    labels = load_labels()
    assert len(labels) == 24
    rows, best = calibrate(scores, labels)
    chosen = next(r for r in rows if r.threshold == best)
    assert chosen.kappa >= 0.8
    misses = disagreements(scores, labels, best, reasons)
    assert any("refund-annual-early" in m and "let through" in m for m in misses)


def test_claude_judge_request_without_network():
    seen = {}

    def fake(url, headers, body):
        seen.update(url=url, body=body)
        return {"content": [{"type": "text", "text": '{"score": 5, "reason": "grounded"}'}]}

    kb = load_kb()
    case = load_cases(kb=kb)[0]
    v = ClaudeJudge(model="judge-model", api_key="test", transport=fake).judge("r", case, kb, "{}")
    assert v.score == 5
    assert seen["body"]["model"] == "judge-model"
    assert "<reply>{}</reply>" in seen["body"]["messages"][0]["content"]


def test_claude_judge_without_key_is_a_clear_error(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "EVAL_JUDGE_MODEL", "EVAL_CLAUDE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(AdapterError):
        ClaudeJudge()


def test_cli_runs_offline(capsys):
    assert main([]) == 0
    assert "<- chosen" in capsys.readouterr().out
