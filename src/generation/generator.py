import re
from typing import List, Optional, Tuple

from src.agent.llm.provider import LLMProvider
from src.prompts import load_prompt


class GroundedGenerator:
    """
    Executes grounded LLM generation strictly conditioned on retrieved context blocks.
    Enforces inline citation tags [DocID#ChunkID] and anti-hallucination guardrails.

    Generation goes through the shared LLMProvider so retries, failover and token
    accounting apply to answers as well as agent calls.
    """

    SYSTEM_PROMPT = load_prompt("system_generator")

    def __init__(self, llm_provider: Optional[LLMProvider] = None, model: Optional[str] = None):
        self.llm = llm_provider or LLMProvider()
        self.model = model or self.llm.model

    async def generate_response(
        self,
        query: str,
        formatted_context: str,
        citation_manifest: List[dict],
    ) -> Tuple[str, List[str]]:
        """
        Generates grounded answer with inline citations.
        Returns: (answer_text, list_of_cited_tags)
        """
        if self.llm.provider_type in ("groq", "openai"):
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"EVIDENCE:\n{formatted_context}\n\nQUESTION: {query}",
                },
            ]
            resp = await self.llm.chat_completion(messages, temperature=0.1, max_tokens=800)
            content = resp.get("content") if resp.get("status") == "success" else None
            if content:
                tags = re.findall(
                    r"[\[\u3010]\s*([a-zA-Z0-9_\-]+#[a-zA-Z0-9_\-]+)\s*[\]\u3011]", content
                )
                return content, [f"[{t}]" for t in dict.fromkeys(tags)]

        # Deterministic Grounded Local Synthesizer (zero-hallucination baseline)
        return self._local_grounded_synthesis(query, formatted_context, citation_manifest)

    def _local_grounded_synthesis(
        self,
        query: str,
        context: str,
        manifest: List[dict],
    ) -> Tuple[str, List[str]]:
        if not manifest:
            return "I cannot confirm this based on the available documentation.", []

        # Find most salient evidence sentences that match the query
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query.lower()))
        best_sentences = []

        for item in manifest:
            tag = item["citation_tag"]
            # Extract evidence block for this tag
            pattern = re.escape(tag) + r".*?\n(.*?)(?=\n---|\Z)"
            match = re.search(pattern, context, re.DOTALL)
            if match:
                evidence_text = match.group(1).strip()
                sentences = re.split(r"(?<=[.!?])\s+", evidence_text)
                for sentence in sentences:
                    candidate = sentence.strip()
                    s_tokens = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", candidate.lower()))
                    overlap = len(q_tokens.intersection(s_tokens))
                    if overlap >= 1 and len(candidate) > 20:
                        best_sentences.append((candidate, tag, overlap))

        if not best_sentences:
            # Fallback to the top candidate excerpt
            top_item = manifest[0]
            answer = f"According to verified records, {top_item['excerpt'].rstrip('.')} {top_item['citation_tag']}."
            return answer, [top_item["citation_tag"]]

        # Sort by relevance overlap
        best_sentences.sort(key=lambda item: item[2], reverse=True)
        selected = best_sentences[:3]

        answer_parts = [f"{sentence.rstrip('.')} {tag}." for sentence, tag, _ in selected]
        cited_tags = list(dict.fromkeys(tag for _, tag, _ in selected))
        return " ".join(answer_parts), cited_tags
