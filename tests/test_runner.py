import json

import pytest

from evalkit.adapters import (AdapterError, ClaudeAdapter, OpenAIAdapter, ReplayAdapter,
                              estimate_tokens, make_adapter)
from evalkit.cases import load_cases, load_kb
from evalkit.pricing import cost_usd, load_prices
from evalkit.runlog import RunLog
from evalkit.runner import main, percentile, run_eval

GOOD = json.dumps({"answer": "Annual plans are refunded in full within 30 days of purchase.",
                   "citations": ["refunds"], "escalate": False})


def test_cost_uses_per_million_prices():
    prices = {"m": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}
    assert cost_usd("m", 1_000_000, 0, prices) == 3.0
    assert cost_usd("m", 1000, 1000, prices) == 0.018
    assert cost_usd("unknown", 1000, 1000, prices) == 0.0


def test_pricing_file_covers_every_built_in_adapter():
    prices = load_prices()
    for model in ("replay:prompt_v1", "replay:prompt_v2", "claude", "openai"):
        assert model in prices


def test_percentile_nearest_rank():
    assert percentile([], 0.5) == 0.0
    assert percentile([10, 20, 30, 40], 0.5) == 20
    assert percentile([10, 20, 30, 40], 0.95) == 40


def test_replay_is_deterministic_and_marked_simulated():
    kb = load_kb()
    case = load_cases(kb=kb)[0]
    a = ReplayAdapter("prompt_v1").generate(case, kb)
    b = ReplayAdapter("prompt_v1").generate(case, kb)
    assert a == b and a.simulated and a.input_tokens > a.output_tokens > 0
    assert estimate_tokens("abcd" * 10) == 10


def test_unknown_adapter_and_missing_keys_fail_clearly(monkeypatch):
    with pytest.raises(AdapterError):
        make_adapter("gemini")
    with pytest.raises(AdapterError):
        ReplayAdapter("no_such_run")
    for var in ("ANTHROPIC_API_KEY", "EVAL_CLAUDE_MODEL", "OPENAI_API_KEY", "EVAL_OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(AdapterError, match="ANTHROPIC_API_KEY"):
        ClaudeAdapter()
    with pytest.raises(AdapterError, match="OPENAI_API_KEY"):
        OpenAIAdapter()


def test_claude_adapter_sends_system_prompt_and_reads_usage():
    seen = {}

    def fake(url, headers, body):
        seen.update(url=url, headers=headers, body=body)
        return {"content": [{"type": "text", "text": GOOD}], "usage": {"input_tokens": 900, "output_tokens": 40}}

    kb = load_kb()
    case = load_cases(kb=kb)[0]
    gen = ClaudeAdapter(model="test-model", api_key="test-key", transport=fake).generate(case, kb)
    assert seen["url"].endswith("/v1/messages")
    assert seen["headers"]["x-api-key"] == "test-key"
    assert "JSON" in seen["body"]["system"]
    assert case.question in seen["body"]["messages"][0]["content"]
    assert (gen.output, gen.model, gen.input_tokens, gen.output_tokens) == (GOOD, "claude", 900, 40)


def test_openai_adapter_asks_for_json_and_reads_usage():
    def fake(url, headers, body):
        assert body["response_format"] == {"type": "json_object"}
        assert headers["authorization"] == "Bearer test-key"
        return {"choices": [{"message": {"content": GOOD}}], "usage": {"prompt_tokens": 800, "completion_tokens": 30}}

    kb = load_kb()
    case = load_cases(kb=kb)[0]
    gen = OpenAIAdapter(model="test-model", api_key="test-key", transport=fake).generate(case, kb)
    assert (gen.model, gen.input_tokens, gen.output_tokens) == ("openai", 800, 30)


def test_run_writes_every_answer_with_cost_and_latency():
    log = RunLog(":memory:")
    run_id, results, summary = run_eval(ReplayAdapter("prompt_v2"), log)
    rows = log.answers(run_id)
    assert len(rows) == len(results) == summary["cases"]
    assert all(r["cost_usd"] > 0 and r["latency_ms"] > 0 and r["simulated"] == 1 for r in rows)
    assert summary["cost_usd"] == pytest.approx(sum(r["cost_usd"] for r in rows), abs=1e-6)
    assert round(sum(r["passed"] for r in rows) / len(rows), 3) == summary["pass_rate"]
    assert log.runs()[0]["finished_at"] is not None


def test_failed_call_is_a_failed_case_not_a_crash():
    class Flaky:
        model = "claude"

        def generate(self, case, kb):
            raise TimeoutError("read timed out")

    log = RunLog(":memory:")
    run_id, results, summary = run_eval(Flaky(), log)
    assert summary["pass_rate"] == 0.0
    assert all("TimeoutError" in r["error"] for r in log.answers(run_id))


def test_cli_compares_two_runs_and_writes_sqlite(tmp_path, capsys):
    db = tmp_path / "runs.sqlite"
    assert main(["replay:prompt_v1", "replay:prompt_v2", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "cost_usd" in out and "p95_ms" in out
    log = RunLog(db)
    assert [r["label"] for r in log.runs()] == ["replay:prompt_v1", "replay:prompt_v2"]
    log.close()


def test_cli_skips_live_adapters_without_keys(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["claude", "--db", ":memory:"]) == 1
    assert "skipped claude" in capsys.readouterr().err
