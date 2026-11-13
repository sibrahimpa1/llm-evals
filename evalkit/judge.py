"""LLM judge, calibrated against hand-labelled answers before anyone trusts it.

The judge scores an answer from 1 to 5 against a rubric. A score alone means nothing until you know
where to put the pass line and how often the judge agrees with a person, so `calibrate` compares
judge scores with hand labels at every threshold and reports accuracy, Cohen's kappa and the misses.

    python -m evalkit.judge                   # offline: recorded judge scores
    python -m evalkit.judge --judge claude    # live: Claude judges the recorded answers again
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .adapters import AdapterError, Transport, http_post_json
from .cases import DATA_DIR, EvalCase, load_cases, load_kb, load_recorded

JUDGE_DIR = DATA_DIR / "judge"
Key = Tuple[str, str]  # (run, case_id)

RUBRIC = (
    "You grade answers from a customer support assistant. You get the help-center documents, the "
    "customer question and the assistant's raw reply. Score 1 to 5:\n"
    "5 correct, grounded in the documents, complete, valid JSON with answer, citations and escalate\n"
    "4 correct and grounded, minor gaps in tone or detail\n"
    "3 mostly right but missing something the customer needs\n"
    "2 wrong hand-off decision, broken format, or a misleading statement\n"
    "1 invents facts, prices, time limits or integrations not in the documents\n"
    'Reply with JSON only: {"score": integer, "reason": string}.'
)


@dataclass
class Verdict:
    score: int
    reason: str


class JudgeError(RuntimeError):
    pass


def parse_verdict(text: str) -> Verdict:
    """Accept the JSON verdict even when a model wraps it in prose or a code fence."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise JudgeError(f"judge reply has no JSON: {text[:80]!r}")
    try:
        data = json.loads(match.group())
        score = int(data["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise JudgeError(f"judge reply is not a verdict: {exc}") from exc
    if not 1 <= score <= 5:
        raise JudgeError(f"judge score {score} is outside 1 to 5")
    return Verdict(score, str(data.get("reason", "")))


class ReplayJudge:
    """Recorded verdicts, so calibration runs offline and gives the same numbers every time."""

    name = "replay"

    def __init__(self, path: Path = JUDGE_DIR / "recorded_scores.jsonl"):
        self.verdicts: Dict[Key, Verdict] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                self.verdicts[(row["run"], row["case_id"])] = Verdict(int(row["score"]), row.get("reason", ""))

    def judge(self, run: str, case: EvalCase, kb: Dict[str, str], output: str) -> Verdict:
        try:
            return self.verdicts[(run, case.id)]
        except KeyError:
            raise JudgeError(f"no recorded verdict for {run}/{case.id}") from None


class ClaudeJudge:
    """Claude as the judge. Model id from EVAL_JUDGE_MODEL (falls back to EVAL_CLAUDE_MODEL)."""

    name = "claude"
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 transport: Transport = http_post_json):
        self.model = model or os.environ.get("EVAL_JUDGE_MODEL") or os.environ.get("EVAL_CLAUDE_MODEL", "")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.model or not self.api_key:
            raise AdapterError("set ANTHROPIC_API_KEY and EVAL_JUDGE_MODEL (see .env.example)")
        self.transport = transport

    def judge(self, run: str, case: EvalCase, kb: Dict[str, str], output: str) -> Verdict:
        docs = "\n\n".join(f"<doc id=\"{d}\">\n{kb[d]}\n</doc>" for d in case.context_docs)
        user = f"{docs}\n\n<question>{case.question}</question>\n\n<reply>{output}</reply>"
        body = {"model": self.model, "max_tokens": 200, "system": RUBRIC,
                "messages": [{"role": "user", "content": user}]}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        resp = self.transport(self.url, headers, body)
        text = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
        return parse_verdict(text)


def load_labels(path: Path = JUDGE_DIR / "labels.jsonl") -> Dict[Key, bool]:
    labels: Dict[Key, bool] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["label"] not in ("pass", "fail"):
            raise ValueError(f"labels line {n}: label must be 'pass' or 'fail'")
        labels[(row["run"], row["case_id"])] = row["label"] == "pass"
    return labels


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    observed = (tp + tn) / n
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    return 0.0 if expected == 1 else round((observed - expected) / (1 - expected), 3)


@dataclass
class ThresholdRow:
    threshold: int
    tp: int  # judge pass, person pass
    fp: int  # judge pass, person fail: the dangerous one, a bad answer let through
    fn: int  # judge fail, person pass
    tn: int
    accuracy: float
    kappa: float


def calibrate(scores: Dict[Key, int], labels: Dict[Key, bool]) -> Tuple[List[ThresholdRow], int]:
    """Agreement at each pass line. The pick is the best kappa; ties go to the stricter line."""
    keys = sorted(set(scores) & set(labels))
    if not keys:
        raise JudgeError("no judged answer has a hand label")
    rows = []
    for t in range(2, 6):
        tp = sum(scores[k] >= t and labels[k] for k in keys)
        fp = sum(scores[k] >= t and not labels[k] for k in keys)
        fn = sum(scores[k] < t and labels[k] for k in keys)
        tn = sum(scores[k] < t and not labels[k] for k in keys)
        rows.append(ThresholdRow(t, tp, fp, fn, tn, round((tp + tn) / len(keys), 3), cohen_kappa(tp, fp, fn, tn)))
    best = max(rows, key=lambda r: (r.kappa, r.threshold))
    return rows, best.threshold


def disagreements(scores: Dict[Key, int], labels: Dict[Key, bool], threshold: int,
                  reasons: Optional[Dict[Key, str]] = None) -> List[str]:
    out = []
    for k in sorted(set(scores) & set(labels)):
        judge_pass = scores[k] >= threshold
        if judge_pass != labels[k]:
            kind = "let through" if judge_pass else "too strict"
            why = (reasons or {}).get(k, "")
            out.append(f"  {k[0]}/{k[1]:<22} judge {scores[k]} ({kind}) {why}".rstrip())
    return out


def judge_runs(judge, runs=("prompt_v1", "prompt_v2")) -> Tuple[Dict[Key, int], Dict[Key, str]]:
    kb = load_kb()
    cases = load_cases(kb=kb)
    scores: Dict[Key, int] = {}
    reasons: Dict[Key, str] = {}
    for run in runs:
        outputs = load_recorded(DATA_DIR / "recorded" / f"{run}.jsonl")
        for case in cases:
            v = judge.judge(run, case, kb, outputs[case.id])
            scores[(run, case.id)] = v.score
            reasons[(run, case.id)] = v.reason
    return scores, reasons


def format_calibration(rows: List[ThresholdRow], best: int) -> str:
    header = f"{'pass if >=':<12}{'accuracy':>10}{'kappa':>8}{'let_through':>13}{'too_strict':>12}"
    lines = [header, "-" * len(header)]
    for r in rows:
        mark = "  <- chosen" if r.threshold == best else ""
        lines.append(f"{r.threshold:<12}{r.accuracy:>10.0%}{r.kappa:>8.2f}{r.fp:>13}{r.fn:>12}{mark}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the LLM judge against hand labels.")
    parser.add_argument("--judge", default="replay", choices=("replay", "claude"))
    args = parser.parse_args(argv)
    try:
        judge = ReplayJudge() if args.judge == "replay" else ClaudeJudge()
    except AdapterError as exc:
        print(f"skipped judge: {exc}", file=sys.stderr)
        return 1
    scores, reasons = judge_runs(judge)
    labels = load_labels()
    rows, best = calibrate(scores, labels)
    print(f"Judge: {judge.name}, {len(set(scores) & set(labels))} hand-labelled answers")
    print(format_calibration(rows, best))
    misses = disagreements(scores, labels, best, reasons)
    print(f"\nDisagreements at pass >= {best}: {len(misses)}")
    print("\n".join(misses) or "  none")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
