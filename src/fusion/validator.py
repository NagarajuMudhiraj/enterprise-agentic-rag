from typing import List, Set, Optional
from src.common.models import RetrievalCandidate

class EvidenceValidator:
    """
    Validates candidates for deduplication, document provenance, and access permissions.
    """

    @staticmethod
    def validate_and_deduplicate(
        candidates: List[RetrievalCandidate], 
        role_filter: Optional[List[str]] = None
    ) -> List[RetrievalCandidate]:
        allowed = set(role_filter or ["public"])
        validated: List[RetrievalCandidate] = []
        seen_texts: Set[str] = set()

        for cand in candidates:
            chunk = cand.chunk
            
            # ACL Security Validation
            if not set(chunk.allowed_roles).intersection(allowed):
                continue

            # Provenance Check: doc_id must exist
            if not chunk.doc_id:
                continue

            # Deduplication: normalized content check
            normalized = " ".join(chunk.text.lower().split()[:25])
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)

            validated.append(cand)

        return validated
