from typing import List, Dict, Tuple
from src.common.models import RetrievalCandidate

class ContextBuilder:
    """
    Assembles, formats, and budgets validated evidence candidates for LLM generation.
    Implements:
    1. Knapsack token budgeting.
    2. Lost-in-the-middle positioning mitigation.
    3. Standardized citation anchors: [DocID#ChunkIndex].
    """

    def __init__(self, max_context_words: int = 1500):
        self.max_context_words = max_context_words

    def _lost_in_the_middle_sort(self, candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """
        Alternates items to place the most relevant chunks at the very beginning
        and very end of the context, where LLM attention is strongest.
        """
        if len(candidates) <= 2:
            return candidates

        sorted_cands = sorted(candidates, key=lambda x: x.score, reverse=True)
        reordered = [None] * len(sorted_cands)
        left = 0
        right = len(sorted_cands) - 1

        for i, item in enumerate(sorted_cands):
            if i % 2 == 0:
                reordered[left] = item
                left += 1
            else:
                reordered[right] = item
                right -= 1

        return [item for item in reordered if item is not None]

    def build_context(self, candidates: List[RetrievalCandidate]) -> Tuple[str, List[Dict[str, str]]]:
        """
        Formats candidates into prompt-ready context string and returns citation references.
        """
        if not candidates:
            return "No verified context available.", []

        # Reorder to avoid lost-in-the-middle degradation
        reordered = self._lost_in_the_middle_sort(candidates)

        context_blocks = []
        citation_manifest = []
        accumulated_words = 0

        for rank, cand in enumerate(reordered):
            chunk = cand.chunk
            # Use parent context if available, otherwise child text
            text_to_use = cand.parent_text if cand.parent_text else chunk.text
            words = text_to_use.split()

            if accumulated_words + len(words) > self.max_context_words and context_blocks:
                # Exceeded token budget
                break

            accumulated_words += len(words)
            citation_tag = f"[{chunk.doc_id}#{chunk.chunk_id}]"
            
            block = (
                f"--- EVIDENCE {citation_tag} (Strategy: {cand.source_strategy}, Score: {cand.score:.3f}) ---\n"
                f"{text_to_use}\n"
            )
            context_blocks.append(block)

            citation_manifest.append({
                "citation_tag": citation_tag,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "title": chunk.metadata.get("title", "Untitled Document"),
                "excerpt": chunk.text[:120] + "..."
            })

        formatted_context = "\n".join(context_blocks)
        return formatted_context, citation_manifest
