import re
from typing import List
from src.common.models import QueryIntent

class QueryDecomposer:
    """
    Decomposes multi-part queries into focused sub-queries and generates
    Hypothetical Document Embeddings (HyDE) expansions.
    """

    @staticmethod
    def decompose(intent: QueryIntent) -> List[str]:
        query = intent.raw_query
        q_lower = query.lower()

        # Handle comparative queries: "Compare X and Y", "X vs Y"
        compare_match = re.search(r'(?:compare|difference between)\s+(.+?)\s+(?:and|with|versus|vs)\s+(.+)', q_lower)
        if compare_match:
            item_a = compare_match.group(1).strip()
            item_b = compare_match.group(2).strip()
            return [
                f"Features and specifications of {item_a}",
                f"Features and specifications of {item_b}",
                query
            ]

        # Handle multi-condition conjunctions ("X as well as Y", "X and also Y")
        if " and also " in q_lower or " as well as " in q_lower:
            parts = re.split(r'\band also\b|\bas well as\b', query, flags=re.IGNORECASE)
            return [p.strip() for p in parts if len(p.strip()) > 5]

        return [query]

    @staticmethod
    def generate_hyde(query: str) -> str:
        """
        Creates an expanded pseudo-passage (HyDE) reflecting the expected answer structure.
        Bridges the vocabulary gap between a brief query and dense index documents.
        """
        clean_q = query.rstrip('?.,')
        hyde_document = (
            f"The documentation regarding {clean_q} specifies the core architecture, "
            f"operational policies, implementation parameters, dependencies, and requirements. "
            f"Specifically, {clean_q} operates in accordance with verified system standards."
        )
        return hyde_document
