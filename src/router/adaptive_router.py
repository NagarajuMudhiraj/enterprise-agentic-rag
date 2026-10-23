from typing import Dict, Any, List
from src.common.models import QueryIntent
from src.router.intent_classifier import QueryIntentClassifier
from src.router.query_decomposer import QueryDecomposer

class AdaptiveRouter:
    """
    Adaptive strategy router that analyzes query intent, decomposes queries,
    and assigns execution weights and active paths across the multi-modal retrieval layer.
    """

    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold

    def route_query(self, query: str) -> Dict[str, Any]:
        intent = QueryIntentClassifier.classify(query)
        sub_queries = QueryDecomposer.decompose(intent)
        intent.sub_queries = sub_queries
        hyde_context = QueryDecomposer.generate_hyde(query)

        # Fail-Safe Rule: If confidence is borderline, ensure Hybrid retrieval is enforced
        if intent.confidence < self.confidence_threshold:
            intent.needs_hybrid = True
            intent.needs_graph = True

        routing_plan = {
            "intent": intent,
            "primary_strategy": intent.intent_type,
            "active_paths": {
                "hybrid": intent.needs_hybrid,
                "graph": intent.needs_graph,
                "deep": intent.needs_deep,
            },
            "sub_queries": sub_queries,
            "hyde_passage": hyde_context,
            "entities": intent.entities,
        }
        return routing_plan
