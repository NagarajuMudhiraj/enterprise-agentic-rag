import re
from typing import List, Dict, Any
from src.agent.llm.provider import LLMProvider
from src.common.config import settings
from src.prompts import load_prompt

class AgentPlanner:
    """
    Analyzes complex user questions and formulates a step-by-step goal plan.
    Identifies multi-hop dependencies, entities, and comparison dimensions.
    """

    SYSTEM_PROMPT = load_prompt("system_planner")

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def generate_plan(self, query: str) -> List[str]:
        # If running with Groq or OpenAI, prompt the LLM for a structured plan
        if self.llm.provider_type in ["groq", "openai"]:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}"}
            ]
            # Planning is lightweight: use the fast model on Groq so the main model's token budget is kept for answering
            plan_model = settings.GROQ_FAST_MODEL if self.llm.provider_type == "groq" else None
            resp = await self.llm.chat_completion(messages, temperature=0.0, max_tokens=512, model=plan_model)
            if resp.get("status") == "success" and resp.get("content"):
                lines = [
                    re.sub(r'^\d+[\.\)]\s*', '', l.strip()) 
                    for l in resp["content"].split("\n") 
                    if l.strip() and re.match(r'^\d+[\.\)]', l.strip())
                ]
                if lines:
                    return lines[:3]

        # Deterministic Rule-Based Planner Fallback
        q_lower = query.lower()
        goals = []

        # Check for comparative queries
        compare_match = re.search(r'(?:compare|difference between)\s+(.+?)\s+(?:and|with|versus|vs)\s+(.+)', q_lower)
        if compare_match:
            a = compare_match.group(1).strip()
            b = compare_match.group(2).strip()
            goals.append(f"Retrieve architecture and specifications for {a}")
            goals.append(f"Retrieve architecture and specifications for {b}")
            goals.append("Synthesize comparison matrix across both components")
            return goals

        STOP_WORDS = {"what", "when", "where", "which", "who", "how", "does", "can", "will", "should", "is", "are", "explain", "describe", "explore", "retrieve", "please", "tell"}
        raw_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_\-\.]{2,}\b', query)
        entities = [e for e in raw_entities if e.lower() not in STOP_WORDS]

        # Check for relational dependency questions
        if any(w in q_lower for w in ["depend", "connect", "require", "relate", "communicate"]):
            if entities:
                goals.append(f"Explore Knowledge Graph relations and dependencies for {entities[0]}")
            goals.append(f"Retrieve policy and technical documentation regarding {query}")
            return goals


        # Standard factoid / lookup
        goals.append(f"Retrieve verified evidence chunks for '{query}'")
        return goals
