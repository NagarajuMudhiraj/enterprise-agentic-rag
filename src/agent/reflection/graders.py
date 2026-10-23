import re
from typing import List, Dict, Any, Tuple
from src.agent.core.scratchpad import AgentScratchpad

class RetrievalGrader:
    """
    Corrective RAG (CRAG) Grader:
    Evaluates whether tool observations contain relevant, actionable information.
    """

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

        q_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower()))
        obs_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', obs_str))

        overlap = len(q_tokens.intersection(obs_tokens))
        ratio = overlap / max(1, len(q_tokens))

        if ratio >= 0.35 or len(obs_tokens) > 40:
            return True, min(1.0, ratio + 0.3), "Observation contains relevant evidence."
        else:
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
