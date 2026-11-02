"""Abstention semantics: a decline must not be reported as unverified claims.

When no retrieved passage supports a question the system declines to answer.
That is not the same outcome as "answered, but the claims are ungrounded", and
the two used to render identically: the refusal text was fed to the claim
verifier, which reported the apology itself as evidence-less claims and a
faithfulness score of 0.20.
"""

import pytest

from src.agent.core.react_engine import ReActAgentEngine
from src.agent.core.scratchpad import AgentScratchpad
from src.agent.llm.provider import LLMProvider
from src.agent.reflection.graders import RetrievalGrader, query_terms
from src.agent.tools.registry import ToolRegistry


def make_engine() -> ReActAgentEngine:
    return ReActAgentEngine(tool_registry=ToolRegistry(), llm_provider=LLMProvider())


def make_scratchpad(query: str, passages) -> AgentScratchpad:
    scratchpad = AgentScratchpad(user_query=query, max_iterations=3)
    scratchpad.gathered_passages.extend(passages)
    return scratchpad


PDA_QUERY = "Explain the working of PDA is there"

MATCHING_PASSAGE = {
    "citation_tag": "[8a68897d#c_8a68897d_1_0]",
    "doc_id": "8a68897d",
    "chunk_id": "c_8a68897d_1_0",
    "text": "Explain the working of a PDA with stack operations (push, pop, no operation).",
    "source": "CO3 Assignment questions.docx",
}

UNRELATED_PASSAGE = {
    "citation_tag": "[cdd75925#c_cdd75925_0_10]",
    "doc_id": "cdd75925",
    "chunk_id": "c_cdd75925_0_10",
    "text": "deep sequence models overfit on the protein dataset.",
    "source": "AD.docx",
}


# ---------------------------------------------------------------- query terms
def test_query_terms_truncate_to_a_shared_stem():
    # "backups" and "backup" must match without a stemmer dependency.
    assert query_terms("backup schedule") & query_terms("automated backups run nightly")


def test_query_terms_drop_function_words():
    # "explain / the / working / there" carry no topic; only the acronym remains.
    assert query_terms("Explain the working of PDA is there") == {"pda"}


# ----------------------------------------------------------- passage grading
def test_passage_relevance_separates_signal_from_noise():
    assert RetrievalGrader.passage_relevance(PDA_QUERY, MATCHING_PASSAGE["text"]) == 1.0
    assert RetrievalGrader.passage_relevance(PDA_QUERY, UNRELATED_PASSAGE["text"]) == 0.0


def test_grade_observation_judges_passages_not_the_json_envelope():
    observation = {"query": PDA_QUERY, "results_count": 1, "passages": [UNRELATED_PASSAGE]}
    is_relevant, confidence, reason = RetrievalGrader.grade_observation(PDA_QUERY, observation)
    assert is_relevant is False
    assert confidence == 0.0
    assert "significant terms" in reason

    relevant_observation = {"passages": [MATCHING_PASSAGE]}
    is_relevant, confidence, _ = RetrievalGrader.grade_observation(PDA_QUERY, relevant_observation)
    assert is_relevant is True
    assert confidence > 0.5


def test_grade_observation_still_scores_plain_strings():
    is_relevant, score, _ = RetrievalGrader.grade_observation(
        "PostgreSQL ledger",
        "PaymentGateway depends on PostgreSQL for ledger transaction logs.",
    )
    assert is_relevant is True
    assert score > 0.5


# ------------------------------------------------------------- abstention text
@pytest.mark.parametrize(
    "answer",
    [
        "I'm unable to locate documentation that defines a PDA.",
        "I cannot confirm this based on the available documentation.",
        "The available documents do not contain that definition.",
    ],
)
def test_abstention_is_recognised(answer):
    assert ReActAgentEngine._is_abstention(answer) is True


def test_a_cited_answer_is_never_treated_as_an_abstention():
    # Citable prose can legitimately describe a missing detail; only an
    # uncited decline counts as abstaining.
    answer = "The retention period is not found in this policy [d1#c1]."
    assert ReActAgentEngine._is_abstention(answer, ["[d1#c1]"]) is False


# ----------------------------------------------------------- verdict contract
def test_abstention_reports_no_claims_and_no_faithfulness_score():
    engine = make_engine()
    scratchpad = make_scratchpad(PDA_QUERY, [MATCHING_PASSAGE])
    answer = "I'm unable to locate documentation that defines a PDA."

    text, grounded, faithfulness, claims, abstained = engine._evaluate_answer(
        PDA_QUERY, answer, [], scratchpad
    )

    assert abstained is True
    assert grounded is False
    assert claims == []            # the apology is not a factual claim
    assert faithfulness == 0.0     # and there is no score to report

    # The refusal now points at what the corpus does contain, so the user can act.
    assert MATCHING_PASSAGE["citation_tag"] in text
    assert "CO3 Assignment questions.docx" in text


def test_abstention_pointer_skips_unrelated_passages():
    engine = make_engine()
    scratchpad = make_scratchpad(PDA_QUERY, [UNRELATED_PASSAGE])

    text, _, _, _, abstained = engine._evaluate_answer(
        PDA_QUERY, "I cannot confirm this based on the available documentation.", [], scratchpad
    )

    assert abstained is True
    assert UNRELATED_PASSAGE["citation_tag"] not in text
    assert "Closest matching passages" not in text


def test_a_real_answer_still_gets_graded():
    engine = make_engine()
    scratchpad = make_scratchpad(PDA_QUERY, [MATCHING_PASSAGE])
    answer = "A PDA uses a stack with push, pop and no-operation moves [8a68897d#c_8a68897d_1_0]."

    text, grounded, faithfulness, claims, abstained = engine._evaluate_answer(
        PDA_QUERY, answer, ["[8a68897d#c_8a68897d_1_0]"], scratchpad
    )

    assert abstained is False
    assert text == answer
    assert faithfulness > 0.0


# --------------------------------------------------------- citation integrity
def test_citations_are_filtered_to_retrieved_passages():
    """A tag the model invented must not become a clickable "evidence" chip."""
    engine = make_engine()
    scratchpad = make_scratchpad(PDA_QUERY, [MATCHING_PASSAGE])

    kept = engine._known_citations(
        [MATCHING_PASSAGE["citation_tag"], "[deadbeef#c_invented]"], scratchpad
    )

    assert kept == [MATCHING_PASSAGE["citation_tag"]]


def test_citations_left_alone_when_nothing_was_retrieved():
    engine = make_engine()
    scratchpad = make_scratchpad(PDA_QUERY, [])

    assert engine._known_citations(["[d1#c1]"], scratchpad) == ["[d1#c1]"]
