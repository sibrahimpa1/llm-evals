# llm-evals: eval and guardrail harness for an AI feature

A test harness for an AI support assistant. It scores every answer for **format**, **grounding** and **accuracy**, runs on every pull request and is built to block a regression before it reaches customers.

All data is synthetic: a made-up help center (refunds, billing, security, integrations) and a hand-written eval set of 12 questions, including ones the assistant should hand to a person.

## Run it

```bash
pip install -e ".[dev]"
python -m evalkit.demo   # offline, no API key
pytest -q
```

Compare models and keep a run log (SQLite, one row per answer with tokens, cost and latency):

```bash
python -m evalkit.runner replay:prompt_v1 replay:prompt_v2 --db runs/evals.sqlite
python -m evalkit.runner claude openai     # live: copy .env.example, set keys and model ids
```

A live adapter without a key is skipped with a message, not a crash. A single failed API call is logged as a failed case with its error, so one timeout does not lose the whole run.

The demo scores two recorded prompt versions. `prompt_v1` passes every case. `prompt_v2` looks fine at a glance, but the harness catches an invented refund time, a broken JSON reply, a deletion request it should have escalated and a made-up Salesforce integration.

## What each scorer checks

| Scorer | Fails when |
|---|---|
| format | the reply is not JSON with `answer`, `citations` and `escalate` of the right types |
| grounding | it cites a document that was not retrieved, or states a number (days, price, limit) that is not in a cited document |
| accuracy | a required fact is missing, a forbidden claim appears, or the hand-off decision is wrong |

The scorers in this version are plain code, not a model. They are free, fast and give the same result every run, which is what a CI gate needs.

## Models, cost and latency

Every adapter sends the same system prompt (`evalkit/prompt.py`) and returns the reply, token counts and latency.

| Adapter | What it does |
|---|---|
| `replay:<name>` | plays back a recorded run from `data/recorded/`; tokens are estimated and latency is simulated, and the log marks those rows as simulated |
| `claude` | Anthropic Messages API, model id from `EVAL_CLAUDE_MODEL` |
| `openai` | OpenAI Chat Completions in JSON mode, model id from `EVAL_OPENAI_MODEL` |

The adapters use the standard library for HTTP, so the harness has no SDK version to keep in step. Prices per million tokens live in `data/pricing.json`; the values there are illustrative and should be set to your provider's current prices before comparing real runs. Latency is reported as p50 and p95, nearest rank, because a 12-case set is too small for interpolated percentiles to mean much.

## Roadmap

1. Eval set, scorers, tests, CI and offline demo (done)
2. Runner with model adapters (offline replay, Claude, a second model) and a run log with cost and latency (this version)
3. Guardrails: prompt-injection and red-team cases, PII-leak and refusal checks, an LLM judge calibrated against hand-labelled cases
4. Regression gate: a committed baseline that fails the pull request when quality drops
5. Dashboard comparing quality, cost and latency per model

## License

MIT
