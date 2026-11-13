"""The prompt the support feature sends to a model. Kept in one place so every adapter sends the same thing."""
from __future__ import annotations

from typing import Dict

from .cases import EvalCase

SYSTEM = (
    "You are a support assistant for a software company. Answer only from the documents you are given. "
    "Reply with JSON and nothing else: "
    '{"answer": string, "citations": [doc ids you used], "escalate": boolean}. '
    "Set escalate to true when the documents do not cover the question, when the customer asks for a person, "
    "or when the request needs a human (account deletion, legal, security incidents). "
    "Never invent prices, time limits or integrations."
)


def build_user_message(case: EvalCase, kb: Dict[str, str]) -> str:
    docs = "\n\n".join(f"<doc id=\"{d}\">\n{kb[d]}\n</doc>" for d in case.context_docs)
    return f"{docs}\n\nCustomer question: {case.question}"
