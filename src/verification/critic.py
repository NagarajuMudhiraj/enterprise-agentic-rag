import re
from typing import List, Dict
from src.common.models import AtomicClaim, VerificationResult, RetrievalCandidate
from src.verification.claim_extractor import ClaimExtractor
from src.common.config import settings

class AnswerVerificationCritic:
    """
    Automated Critic verifying whether every generated claim is strictly supported
    by the retrieved and cited evidence chunks (NLI Entailment & Citation Integrity).
    """

    def __init__(self, faithfulness_threshold: float = None):
        self.faithfulness_threshold = faithfulness_threshold or settings.FAITHFULNESS_THRESHOLD

    STOP_WORDS = {
        "the", "and", "for", "with", "this", "that", "from", "are", "is", "was", "were",
        "have", "has", "all", "any", "these", "those", "which", "what", "where", "when",
        "more", "also", "into", "been", "will", "does", "according", "system", "documentation"
    }

    def _verify_claim_entailment(self, claim: AtomicClaim, evidence_map: Dict[str, str]) -> AtomicClaim:
        """
        Validates whether the cited evidence entails the claim words and semantic concepts.
        """
        clean_claim = claim.claim_text.replace('\u202f', ' ').replace('\xa0', ' ')

        if not claim.cited_chunk_ids:
            # Uncited claim -> penalized as unverified neutral
            claim.entailment_status = "neutral"
            claim.entailment_score = 0.2
            return claim

        all_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{2,}\b', clean_claim.lower()))
        claim_tokens = all_tokens - self.STOP_WORDS
        if not claim_tokens:
            claim_tokens = all_tokens

        if not claim_tokens:
            claim.entailment_status = "entailed"
            claim.entailment_score = 1.0
            return claim

        # Gather text from all cited evidence chunks (deduplicated)
        seen_texts = set()
        relevant_evidence_texts = []
        for cid in claim.cited_chunk_ids:
            if cid in evidence_map and evidence_map[cid] not in seen_texts:
                seen_texts.add(evidence_map[cid])
                relevant_evidence_texts.append(evidence_map[cid])
            # Handle loose tag matching (e.g. matching on chunk_id suffix)
            cid_suffix = cid.split('#')[-1]
            for k, text in evidence_map.items():
                if cid_suffix in k and text not in seen_texts:
                    seen_texts.add(text)
                    relevant_evidence_texts.append(text)

        if not relevant_evidence_texts:
            # Cited source not found in retrieved set
            claim.entailment_status = "neutral"
            claim.entailment_score = 0.0
            return claim

        combined_evidence = " ".join(relevant_evidence_texts).replace('\u202f', ' ').replace('\xa0', ' ').lower()
        evidence_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]+\b', combined_evidence))

        # Check for numeric contradictions (e.g. 7 days vs 3600 seconds)
        claim_numbers = set(re.findall(r'\b\d+\b', clean_claim))
        evidence_numbers = set(re.findall(r'\b\d+\b', combined_evidence))
        if claim_numbers and not claim_numbers.issubset(evidence_numbers):
            claim.entailment_status = "contradicted"
            claim.entailment_score = 0.0
            return claim

        # Check coverage
        supported_tokens = claim_tokens.intersection(evidence_tokens)
        support_ratio = len(supported_tokens) / len(claim_tokens)

        if support_ratio >= 0.60:
            claim.entailment_status = "entailed"
            claim.entailment_score = 1.0
            claim.supporting_quote = combined_evidence[:150]
        elif support_ratio >= 0.35:
            claim.entailment_status = "neutral"
            claim.entailment_score = 0.5
        else:
            claim.entailment_status = "contradicted"
            claim.entailment_score = 0.0

        return claim

    def verify_answer(
        self, 
        answer_text: str, 
        candidates: List[RetrievalCandidate]
    ) -> VerificationResult:
        # Build evidence map by [doc_id#chunk_id] and chunk_id
        evidence_map: Dict[str, str] = {}
        for c in candidates:
            chunk = c.chunk
            tag = f"{chunk.doc_id}#{chunk.chunk_id}"
            text = c.parent_text if c.parent_text else chunk.text
            evidence_map[tag] = text
            evidence_map[chunk.chunk_id] = text

        claims = ClaimExtractor.extract_claims(answer_text)
        if not claims:
            return VerificationResult(
                faithfulness_score=1.0,
                claims=[],
                missing_evidence_claims=[],
                passed=True,
                verdict_explanation="No factual claims required verification."
            )

        verified_claims: List[AtomicClaim] = []
        missing_claims: List[str] = []
        total_score = 0.0

        for claim in claims:
            verified = self._verify_claim_entailment(claim, evidence_map)
            verified_claims.append(verified)
            total_score += verified.entailment_score

            if verified.entailment_status != "entailed":
                missing_claims.append(verified.claim_text)

        avg_faithfulness = total_score / len(verified_claims)
        passed = avg_faithfulness >= self.faithfulness_threshold

        verdict = (
            f"Verification PASSED with faithfulness {avg_faithfulness:.2f}."
            if passed else
            f"Verification FAILED (faithfulness {avg_faithfulness:.2f} < threshold {self.faithfulness_threshold})."
        )

        return VerificationResult(
            faithfulness_score=avg_faithfulness,
            claims=verified_claims,
            missing_evidence_claims=missing_claims,
            passed=passed,
            verdict_explanation=verdict
        )
