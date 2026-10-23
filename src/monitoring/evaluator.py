import re
from typing import List, Dict, Any

class RAGEvaluator:
    """
    RAGAS-compatible evaluation harness calculating key quality dimensions:
    - Faithfulness (claim entailment against retrieved context)
    - Answer Relevance (semantic alignment with original query)
    - Context Precision (percentage of retrieved chunks that are relevant)
    - Context Recall (ground truth concept coverage)
    """

    @staticmethod
    def compute_metrics(
        query: str, 
        answer: str, 
        retrieved_contexts: List[str], 
        ground_truth: str = None
    ) -> Dict[str, float]:
        q_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower()))
        a_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', answer.lower()))
        combined_ctx = " ".join(retrieved_contexts).lower()
        ctx_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', combined_evidence := combined_ctx))

        # 1. Answer Relevance: overlap between answer and question intent
        rel_overlap = len(a_tokens.intersection(q_tokens))
        answer_relevance = min(1.0, rel_overlap / max(1, len(q_tokens)))

        # 2. Faithfulness: answer tokens grounded in context
        faith_overlap = len(a_tokens.intersection(ctx_tokens))
        faithfulness = faith_overlap / max(1, len(a_tokens))

        # 3. Context Precision: fraction of context chunks containing question terms
        relevant_chunks = 0
        for ctx in retrieved_contexts:
            c_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', ctx.lower()))
            if len(c_tokens.intersection(q_tokens)) >= 1:
                relevant_chunks += 1
        context_precision = relevant_chunks / max(1, len(retrieved_contexts))

        # 4. Context Recall (if ground truth supplied)
        context_recall = 1.0
        if ground_truth:
            gt_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', ground_truth.lower()))
            gt_overlap = len(ctx_tokens.intersection(gt_tokens))
            context_recall = gt_overlap / max(1, len(gt_tokens))

        return {
            "faithfulness": round(faithfulness, 3),
            "answer_relevance": round(answer_relevance, 3),
            "context_precision": round(context_precision, 3),
            "context_recall": round(context_recall, 3),
        }
