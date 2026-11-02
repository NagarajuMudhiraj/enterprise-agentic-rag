import re
from typing import List, Dict, Any, Tuple
from src.agent.core.scratchpad import AgentScratchpad

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[\-\.][a-z0-9]+)*")

# Function words carry no topical signal; keeping them let unrelated passages
# score as relevant purely because a question mark and a "the" overlapped.
_STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was", "were",
    "have", "has", "not", "but", "you", "your", "can", "could", "would",
    "should", "what", "when", "where", "which", "who", "whom", "how", "does",
    "did", "explain", "describe", "tell", "please", "any", "all", "about",
    "into", "its", "their", "there", "here", "then", "than", "them", "these",
    "those", "been", "being", "will", "also", "more", "most", "some", "such",
    "only", "other", "over", "very", "just", "because", "while", "after",
    "before", "between", "use", "using", "work", "works", "working",
}


def query_terms(text: str) -> set:
    """Significant terms of a query or passage, truncated to a 4-character stem.

    Truncation keeps inflectional variants together ("backups" / "backup",
    "scheduled" / "schedule") without pulling in a stemmer dependency.
    """
    return {
        tok[:4]
        for tok in _TOKEN_RE.findall((text or "").lower())
        if len(tok) >= 3 and tok not in _STOP_WORDS
    }


class RetrievalGrader:
    """
    Corrective RAG (CRAG) Grader:
    Evaluates whether tool observations contain relevant, actionable information.
    """

    @staticmethod
    def passage_relevance(query: str, passage_text: str) -> float:
        """Share of the query's significant terms that appear in one passage (0..1)."""
        q_terms = query_terms(query)
        if not q_terms:
            return 0.0
        return len(q_terms & query_terms(passage_text)) / len(q_terms)

    @staticmethod
    def grade_observation(query: str, observation: Any) -> Tuple[bool, float, str]:
        """
        Returns (is_relevant, confidence_score, feedback_reason)
        """
        if not observation:
            return False, 0.0, "Tool returned empty observation."

        obs_str = str(observation).lower()
        if "error" in obs_str or "not found" in obs_str:
            return False, 0.0, "Tool reported missing entity or search failure."

        # Retrieval tools return a passage list: judge the best passage against the
        # query. Judging the JSON envelope instead used to mark any large blob as
        # relevant, so unrelated passages were badged "✓ Relevant".
        passages = observation.get("passages") if isinstance(observation, dict) else None
        if isinstance(passages, list) and passages:
            best = max(
                (
                    RetrievalGrader.passage_relevance(query, p.get("text", ""))
                    for p in passages
                    if isinstance(p, dict)
                ),
                default=0.0,
            )
            if best >= 0.34:
                return True, round(min(1.0, 0.3 + best), 2), "A retrieved passage shares the query's key terms."
            return (
                False,
                round(best, 2),
                "No retrieved passage shares significant terms with the query.",
            )

        q_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower()))
        obs_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', obs_str))

        overlap = len(q_tokens.intersection(obs_tokens))
        ratio = overlap / max(1, len(q_tokens))

        if ratio >= 0.35:
            return True, min(1.0, ratio + 0.3), "Observation contains relevant evidence."
        return False, ratio, "Observation exhibits low semantic relevance to the query."

class HallucinationGrader:
    """
    Self-RAG Grader:
    Validates whether the synthesized answer's claims are fully entailed by gathered evidence chunks.
    """

    @staticmethod
    def grade_grounding(answer: str, scratchpad: AgentScratchpad) -> Tuple[bool, float, List[str]]:
        """
        Returns (is_grounded, faithfulness_score, missing_claims)
        """
        # Collect all raw text gathered in scratchpad
        corpus_texts = []
        for p in scratchpad.gathered_passages:
            corpus_texts.append(p.get("text", ""))
        for step in scratchpad.steps:
            if step.observation:
                corpus_texts.append(str(step.observation))

        all_evidence = " ".join(corpus_texts).lower()
        if not all_evidence:
            return False, 0.0, ["No evidence was gathered in the scratchpad."]

        STOP_WORDS = {
            "these", "three", "two", "one", "the", "and", "for", "with", "this", "that", 
            "from", "are", "is", "was", "were", "have", "has", "all", "any", "constitute", 
            "external", "component", "complete", "documented", "following", "such", "than", 
            "other", "their", "which", "what", "where", "when", "more", "also", "into", 
            "been", "will", "does", "services", "service", "system", "evidence", "table",
            "relationship", "type", "dependent"
        }
        evidence_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', all_evidence))

        # Split by sentences or table newlines
        raw_units = re.split(r'(?<=[.!?])\s+|\n+', answer)
        units = []
        for u in raw_units:
            clean = u.strip().strip('|').strip()
            if len(clean) > 15 and not re.match(r'^[-:| ]+$', clean):
                units.append(clean)

        if not units:
            return True, 1.0, []

        unsupported = []
        scores = []

        for unit in units:
            unit_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', unit.lower())) - STOP_WORDS
            if not unit_tokens:
                continue
            overlap = len(unit_tokens.intersection(evidence_tokens))
            claim_score = overlap / max(1, len(unit_tokens))
            scores.append(claim_score)
            if claim_score < 0.40:
                unsupported.append(unit)

        avg_score = sum(scores) / len(scores) if scores else 1.0
        has_valid_citations = bool(re.search(r'[\[\u3010]\s*[a-zA-Z0-9_\-]+#[a-zA-Z0-9_\-]+\s*[\]\u3011]', answer))
        # Grounded if overall content overlaps evidence, supported by citations
        is_grounded = (avg_score >= 0.45) or (has_valid_citations and avg_score >= 0.35)

        return is_grounded, round(avg_score, 2), unsupported
