import re
from typing import List, Tuple
from src.common.models import QueryIntent

class QueryIntentClassifier:
    """
    Analyzes user queries to determine search intent, entities, and required retrieval routes.
    """

    RELATIONAL_INDICATORS = [
        "connect", "depend", "relate", "relationship", "difference between", 
        "compare", "versus", "vs", "interaction", "link", "hierarchy", "architecture"
    ]

    SYNTHESIS_INDICATORS = [
        "summarize", "overview", "comprehensive", "explain all", "broad", 
        "architecture", "best practices", "guide", "walkthrough", "full policy"
    ]

    @classmethod
    def classify(cls, query: str) -> QueryIntent:
        q_lower = query.lower()
        
        # Extract potential entities (capitalized phrases or alphanumeric identifiers)
        entities = re.findall(r'\b[A-Z][a-zA-Z0-9_\-\.]{2,}\b', query)
        if not entities:
            # Fallback to key nouns
            entities = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', query) if w.lower() not in ["what", "when", "where", "which", "how", "does", "explain", "describe"]]

        is_relational = any(ind in q_lower for ind in cls.RELATIONAL_INDICATORS)
        is_synthesis = any(ind in q_lower for ind in cls.SYNTHESIS_INDICATORS)

        if is_relational or len(entities) >= 2:
            intent_type = "relational"
            needs_graph = True
            needs_deep = False
            confidence = 0.88
        elif is_synthesis:
            intent_type = "synthesized_summary"
            needs_graph = True
            needs_deep = True
            confidence = 0.85
        else:
            intent_type = "factoid"
            needs_graph = False
            needs_deep = False
            confidence = 0.92

        return QueryIntent(
            raw_query=query,
            intent_type=intent_type,
            sub_queries=[query],
            entities=list(set(entities)),
            needs_hybrid=True, # Always maintain hybrid foundation
            needs_graph=needs_graph,
            needs_deep=needs_deep,
            confidence=confidence
        )
