import re
from typing import List, Optional
from src.common.models import VerificationResult

class FeedbackLoopOrchestrator:
    """
    Manages the closed-loop feedback mechanism:
    - Analyzes verification failures.
    - Generates targeted reformulated queries targeting missing claims.
    - Triggers graceful abstention when retries or confidence bounds are exhausted.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def should_retry(self, attempt: int, verification: VerificationResult) -> bool:
        if verification.passed:
            return False
        return attempt < self.max_retries

    def reformulate_query(self, original_query: str, missing_claims: List[str]) -> str:
        """
        Creates a targeted secondary query to retrieve missing evidence.
        """
        if not missing_claims:
            return original_query

        # Extract focal terms from the first unsupported claim
        unsupported = missing_claims[0].replace('\u202f', ' ').replace('\xa0', ' ')
        keywords = re.findall(r'\b[A-Za-z0-9_\-]{2,}\b', unsupported)
        stop_words = {
            "according", "verified", "system", "documentation", "records", "cannot", "confirm",
            "the", "and", "for", "with", "this", "that", "from", "are", "is", "was", "were",
            "have", "has", "all", "any", "these", "those", "which", "what", "where", "when",
            "into", "also", "more", "been", "will", "does"
        }
        salient_keywords = [k for k in keywords if k.lower() not in stop_words]

        if salient_keywords:
            refined_terms = " ".join(salient_keywords[:4])
            return f"{original_query} specifically regarding {refined_terms}"
        
        return f"{original_query} detailed specification"

    def create_abstention_response(self, query: str, attempt: int) -> str:
        return (
            "I cannot confirm an answer to this question based on the available documentation. "
            f"The verification critic found insufficient evidence across {attempt + 1} retrieval cycles."
        )
