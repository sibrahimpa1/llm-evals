import json

from evalkit.cases import EvalCase
from evalkit.scorers import evaluate_case, parse_output, score_accuracy, score_grounding

KB = {
    "refunds": "Annual plans are refunded in full within 30 days. Refunds arrive in 5 to 7 business days.",
    "billing": "The Team plan costs 29 EUR per seat per month.",
}
CASE = EvalCase(
    id="c1",
    question="Refund?",
    context_docs=["refunds"],
    must_include=["30 days", "in full"],
    must_not_include=["not refunded"],
    expect_escalate=False,
)


def out(answer, citations=("refunds",), escalate=False):
    return json.dumps({"answer": answer, "citations": list(citations), "escalate": escalate})


def test_parse_rejects_non_json_and_missing_keys():
    assert parse_output("hello")[0] is None
    data, problems = parse_output(json.dumps({"answer": "x"}))
    assert data is None and "missing key 'citations'" in problems


def test_parse_rejects_wrong_types():
    data, problems = parse_output(json.dumps({"answer": "x", "citations": "refunds", "escalate": "no"}))
    assert data is None
    assert "'citations' should be list" in problems and "'escalate' should be bool" in problems


def test_good_answer_passes_everything():
    r = evaluate_case(out("Refunded in full within 30 days, arriving in 5 to 7 business days."), CASE, KB)
    assert r.passed, {k: v.reasons for k, v in r.scores.items()}


def test_invented_figure_fails_grounding():
    s = score_grounding(json.loads(out("Refunded in full within 45 days.")), CASE, KB)
    assert not s.passed
    assert "45" in s.reasons[0]


def test_citing_a_doc_that_was_not_retrieved_fails_grounding():
    s = score_grounding(json.loads(out("29 EUR", citations=["billing"])), CASE, KB)
    assert not s.passed
    assert "not retrieved" in s.reasons[0]


def test_figures_without_citations_fail_grounding():
    s = score_grounding(json.loads(out("Within 30 days.", citations=[])), CASE, KB)
    assert not s.passed


def test_no_figures_no_citations_is_grounded():
    s = score_grounding(json.loads(out("Let me check with a colleague.", citations=[])), CASE, KB)
    assert s.passed and s.value == 1.0


def test_accuracy_partial_credit_for_missing_fact():
    s = score_accuracy(json.loads(out("Refunded within 30 days.")), CASE)
    assert not s.passed and s.value == 0.5


def test_forbidden_claim_zeroes_accuracy():
    s = score_accuracy(json.loads(out("30 days, in full. Monthly plans are not refunded.")), CASE)
    assert s.value == 0.0 and not s.passed


def test_wrong_escalation_fails_accuracy():
    s = score_accuracy(json.loads(out("Refunded in full within 30 days.", escalate=True)), CASE)
    assert not s.passed and "escalation wrong" in s.reasons[0]


def test_broken_format_skips_other_scorers():
    r = evaluate_case("Refunds take 30 days.", CASE, KB)
    assert not r.passed
    assert r.scores["grounding"].reasons == ["not scored: output broke the format"]
