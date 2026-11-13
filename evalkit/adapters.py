"""Model adapters. Each one turns an eval case into one model reply plus token counts and latency.

- ReplayAdapter reads recorded outputs, so tests, CI and the demo run offline and cost nothing.
- ClaudeAdapter and OpenAIAdapter call the real APIs with the standard library only (no SDK pins).
  Their HTTP call goes through a `transport` function, so tests can check the request without a network.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from .cases import DATA_DIR, EvalCase, load_recorded
from .prompt import SYSTEM, build_user_message

Transport = Callable[[str, Dict[str, str], dict], dict]


@dataclass
class Generation:
    output: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    simulated: bool = False


class AdapterError(RuntimeError):
    pass


def estimate_tokens(text: str) -> int:
    """Rough count (about 4 characters per token). Only used where the provider gives no real count."""
    return max(1, math.ceil(len(text) / 4))


def http_post_json(url: str, headers: Dict[str, str], body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={**headers, "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https endpoints)
        return json.loads(resp.read().decode("utf-8"))


class ReplayAdapter:
    """Plays back a recorded run. Latency is simulated from a hash so it is stable between runs."""

    def __init__(self, name: str, recorded_dir: Path = DATA_DIR / "recorded"):
        self.name = name
        self.model = f"replay:{name}"
        path = Path(recorded_dir) / f"{name}.jsonl"
        if not path.exists():
            raise AdapterError(f"no recorded run named {name!r} in {recorded_dir}")
        self.outputs = load_recorded(path)

    def generate(self, case: EvalCase, kb: Dict[str, str]) -> Generation:
        if case.id not in self.outputs:
            raise AdapterError(f"{self.name}: no recorded output for case {case.id!r}")
        output = self.outputs[case.id]
        prompt = SYSTEM + build_user_message(case, kb)
        seed = int(hashlib.sha256(f"{self.name}:{case.id}".encode()).hexdigest()[:8], 16)
        latency = 400 + (seed % 900) + 2 * len(output)
        return Generation(output, self.model, estimate_tokens(prompt), estimate_tokens(output),
                          float(latency), simulated=True)


class ClaudeAdapter:
    """Anthropic Messages API. The model id comes from EVAL_CLAUDE_MODEL."""

    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 transport: Transport = http_post_json, max_tokens: int = 600):
        self.model = model or os.environ.get("EVAL_CLAUDE_MODEL", "")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.model or not self.api_key:
            raise AdapterError("set ANTHROPIC_API_KEY and EVAL_CLAUDE_MODEL (see .env.example)")
        self.transport = transport
        self.max_tokens = max_tokens

    def generate(self, case: EvalCase, kb: Dict[str, str]) -> Generation:
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": SYSTEM,
                "messages": [{"role": "user", "content": build_user_message(case, kb)}]}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        start = time.perf_counter()
        resp = self.transport(self.url, headers, body)
        latency = (time.perf_counter() - start) * 1000
        text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
        usage = resp.get("usage", {})
        return Generation(text, "claude", int(usage.get("input_tokens", 0)),
                          int(usage.get("output_tokens", 0)), round(latency, 1))


class OpenAIAdapter:
    """OpenAI Chat Completions API, as the second model to compare against. Model id from EVAL_OPENAI_MODEL."""

    url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 transport: Transport = http_post_json):
        self.model = model or os.environ.get("EVAL_OPENAI_MODEL", "")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.model or not self.api_key:
            raise AdapterError("set OPENAI_API_KEY and EVAL_OPENAI_MODEL (see .env.example)")
        self.transport = transport

    def generate(self, case: EvalCase, kb: Dict[str, str]) -> Generation:
        body = {"model": self.model, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": build_user_message(case, kb)}]}
        headers = {"authorization": f"Bearer {self.api_key}"}
        start = time.perf_counter()
        resp = self.transport(self.url, headers, body)
        latency = (time.perf_counter() - start) * 1000
        text = resp["choices"][0]["message"]["content"] or ""
        usage = resp.get("usage", {})
        return Generation(text, "openai", int(usage.get("prompt_tokens", 0)),
                          int(usage.get("completion_tokens", 0)), round(latency, 1))


def make_adapter(spec: str):
    """'replay:prompt_v1', 'claude' or 'openai'."""
    kind, _, arg = spec.partition(":")
    if kind == "replay":
        return ReplayAdapter(arg or "prompt_v1")
    if kind == "claude":
        return ClaudeAdapter(model=arg or None)
    if kind == "openai":
        return OpenAIAdapter(model=arg or None)
    raise AdapterError(f"unknown adapter {spec!r}: use replay:<name>, claude or openai")
