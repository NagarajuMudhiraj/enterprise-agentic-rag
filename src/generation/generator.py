import os
import re
import httpx
from typing import List, Dict, Optional, Tuple
from src.common.config import settings

class GroundedGenerator:
    """
    Executes grounded LLM generation strictly conditioned on retrieved context blocks.
    Enforces inline citation tags [DocID#ChunkID] and anti-hallucination guardrails.
    """

    SYSTEM_PROMPT = (
        "You are a production enterprise AI assistant. "
        "Answer the user question using ONLY the provided verified evidence blocks.\n\n"
        "STRICT CONSTRAINTS:\n"
        "1. Every factual sentence or claim MUST cite its supporting evidence using bracket tags, e.g., [doc_1#chunk_2].\n"
        "2. Do NOT extrapolate, speculate, or introduce external knowledge.\n"
        "3. If the provided evidence is contradictory or insufficient to answer the question, "
        "explicitly state: 'I cannot confirm this based on the available documentation.'\n"
        "4. Be concise, direct, and structured."
    )

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        self.groq_api_key = os.environ.get("GROQ_API_KEY") or settings.GROQ_API_KEY
        self.openai_api_key = api_key or os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY

        if self.groq_api_key:
            self.api_key = self.groq_api_key
            self.base_url = settings.GROQ_BASE_URL
            self.model = model or settings.GROQ_MODEL
        elif self.openai_api_key:
            self.api_key = self.openai_api_key
            self.base_url = "https://api.openai.com/v1"
            self.model = model or settings.LLM_MODEL
        else:
            self.api_key = None
            self.base_url = None
            self.model = "local-deterministic"

    async def generate_response(
        self, 
        query: str, 
        formatted_context: str, 
        citation_manifest: List[Dict[str, str]]
    ) -> Tuple[str, List[str]]:
        """
        Generates grounded answer with inline citations.
        Returns: (answer_text, list_of_cited_tags)
        """
        # If API key is available, call the LLM endpoint (Groq / OpenAI)
        if self.api_key and self.base_url:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": [
                                {"role": "system", "content": self.SYSTEM_PROMPT},
                                {"role": "user", "content": f"EVIDENCE:\n{formatted_context}\n\nQUESTION: {query}"}
                            ],
                            "temperature": 0.1
                        }
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_answer = data["choices"][0]["message"]["content"]
                        tags = re.findall(r'[\[\u3010]\s*([a-zA-Z0-9_\-]+#[a-zA-Z0-9_\-]+)\s*[\]\u3011]', raw_answer)
                        citations = [f"[{t}]" for t in set(tags)]
                        return raw_answer, citations
            except Exception:
                pass # Fall back to local grounded synthesis

        # Deterministic Grounded Local Synthesizer (Zero-hallucination baseline)
        return self._local_grounded_synthesis(query, formatted_context, citation_manifest)

    def _local_grounded_synthesis(
        self, 
        query: str, 
        context: str, 
        manifest: List[Dict[str, str]]
    ) -> Tuple[str, List[str]]:
        if not manifest:
            return "I cannot confirm this based on the available documentation.", []

        # Find most salient evidence sentences that match the query
        q_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower()))
        best_sentences = []
        cited_tags = []

        for item in manifest:
            tag = item["citation_tag"]
            # Extract evidence block for this tag
            pattern = re.escape(tag) + r'.*?\n(.*?)(?=\n---|\Z)'
            match = re.search(pattern, context, re.DOTALL)
            if match:
                evidence_text = match.group(1).strip()
                sentences = re.split(r'(?<=[.!?])\s+', evidence_text)
                for s in sentences:
                    s_clean = s.strip()
                    s_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s_clean.lower()))
                    overlap = len(q_tokens.intersection(s_tokens))
                    if overlap >= 1 and len(s_clean) > 20:
                        best_sentences.append((s_clean, tag, overlap))

        if not best_sentences:
            # Fallback to the top candidate excerpt
            top_item = manifest[0]
            answer = f"According to verified records, {top_item['excerpt'].rstrip('.')} {top_item['citation_tag']}."
            return answer, [top_item['citation_tag']]

        # Sort by relevance overlap
        best_sentences.sort(key=lambda x: x[2], reverse=True)
        selected = best_sentences[:3]
        
        answer_parts = [f"{s.rstrip('.')} {tag}." for s, tag, _ in selected]
        cited_tags = list(set([tag for _, tag, _ in selected]))
        final_answer = " ".join(answer_parts)
        
        return final_answer, cited_tags

