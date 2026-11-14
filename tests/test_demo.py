from evalkit.demo import main, run_recorded
from evalkit.report import summarize


def test_v1_passes_and_v2_regresses():
    v1 = summarize(run_recorded("prompt_v1"))
    v2 = summarize(run_recorded("prompt_v2"))
    assert v1["pass_rate"] == 1.0
    assert v2["pass_rate"] < v1["pass_rate"]
    assert v2["grounding"] < 1.0 and v2["format"] < 1.0


def test_demo_runs_offline(capsys):
    assert main([]) == 0
    printed = capsys.readouterr().out
    assert "prompt_v1" in printed and "prompt_v2" in printed
    assert "figures not found in cited docs" in printed
    assert "replay:prompt_v1      PASS" in printed
    assert "replay:prompt_v2      FAIL (11 regressions)" in printed
