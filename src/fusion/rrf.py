from typing import List, Dict
from src.common.models import RetrievalCandidate

class ReciprocalRankFusion:
    """
    Implements Reciprocal Rank Fusion (RRF) to merge candidate lists from
    different retrieval strategies without requiring raw score normalization.
    """

    @staticmethod
    def fuse(
        candidate_lists: List[List[RetrievalCandidate]], 
        k: int = 60,
        top_n: int = 25
    ) -> List[RetrievalCandidate]:
        """
        RRF Score(d) = sum(1 / (k + rank_m(d))) for all lists m where document d appears.
        """
        rrf_table: Dict[str, Dict] = {}

        for cand_list in candidate_lists:
            for rank, candidate in enumerate(cand_list):
                cid = candidate.chunk.chunk_id
                if cid not in rrf_table:
                    rrf_table[cid] = {
                        "candidate": candidate,
                        "rrf_score": 0.0,
                        "contributing_strategies": []
                    }
                # RRF calculation (1-indexed rank)
                rrf_table[cid]["rrf_score"] += 1.0 / (k + rank + 1)
                rrf_table[cid]["contributing_strategies"].append(candidate.source_strategy)

        # Sort descending by fused RRF score
        fused = sorted(rrf_table.values(), key=lambda x: x["rrf_score"], reverse=True)

        final_candidates: List[RetrievalCandidate] = []
        for rank_idx, item in enumerate(fused[:top_n]):
            cand = item["candidate"]
            strategies_joined = "+".join(set(item["contributing_strategies"]))
            
            # Create fused candidate representation
            fused_cand = RetrievalCandidate(
                chunk=cand.chunk,
                source_strategy=f"fused[{strategies_joined}]",
                score=item["rrf_score"],
                rank=rank_idx + 1,
                parent_text=cand.parent_text
            )
            final_candidates.append(fused_cand)

        return final_candidates
