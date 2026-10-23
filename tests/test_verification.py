import pytest
from src.verification.claim_extractor import ClaimExtractor
from src.verification.critic import AnswerVerificationCritic
from src.verification.feedback_loop import FeedbackLoopOrchestrator
from src.common.models import Chunk, RetrievalCandidate

def test_claim_extractor_unicode_and_fullwidth():
    answer = "The database uses port 5432【doc_1#c_1】. The cache server is Redis [ doc_2#c_2 ]."
    claims = ClaimExtractor.extract_claims(answer)
    assert len(claims) == 2
    assert "doc_1#c_1" in claims[0].cited_chunk_ids
    assert "doc_2#c_2" in claims[1].cited_chunk_ids
    assert "5432" in claims[0].claim_text
    assert "Redis" in claims[1].claim_text

def test_claim_extractor_narrow_nobrk_spaces():
    answer = "The key is\u202f12345\u202f[ doc_3#c_3 ]."
    claims = ClaimExtractor.extract_claims(answer)
    assert len(claims) == 1
    assert "doc_3#c_3" in claims[0].cited_chunk_ids
    assert "12345" in claims[0].claim_text

def test_critic_fullwidth_entailment():
    critic = AnswerVerificationCritic()
    chunk = Chunk(
        chunk_id="c_100",
        doc_id="doc_auth",
        text="All auth tokens expire after 3600 seconds."
    )
    cand = RetrievalCandidate(chunk=chunk, source_strategy="hybrid", final_score=1.0)
    
    # Passing claim with fullwidth citation
    answer_pass = "Auth tokens expire in 3600 seconds【doc_auth#c_100】."
    res = critic.verify_answer(answer_pass, [cand])
    assert res.passed is True
    assert res.faithfulness_score >= 0.85

    # Contradicted / missing claim
    answer_fail = "Auth tokens expire in 7 days【doc_auth#c_100】."
    res_fail = critic.verify_answer(answer_fail, [cand])
    assert res_fail.passed is False

def test_feedback_loop_reformulation():
    orchestrator = FeedbackLoopOrchestrator(max_retries=2)
    missing = ["PaymentGateway requires VaultService for tokenized master keys."]
    reformulated = orchestrator.reformulate_query("What does PaymentGateway require?", missing)
    assert "PaymentGateway" in reformulated or "VaultService" in reformulated
    assert reformulated != "What does PaymentGateway require?"
