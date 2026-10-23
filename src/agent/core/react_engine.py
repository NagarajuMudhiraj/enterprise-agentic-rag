import json
import time
import re
from typing import List, Dict, Any, Optional
from src.agent.tools.registry import ToolRegistry
from src.agent.core.scratchpad import AgentScratchpad
from src.agent.core.planner import AgentPlanner
from src.agent.llm.provider import LLMProvider
from src.agent.events.event_bus import AgentEventBus
from src.agent.reflection.graders import RetrievalGrader, HallucinationGrader
from src.common.models import VerifiedResponse
from src.monitoring.security import SecurityManager

class ReActAgentEngine:
    """
    Autonomous ReAct (Reasoning + Action) & Plan-and-Solve Agent Core.
    Dynamically plans sub-goals, invokes tools via function calling,
    evaluates observations (CRAG), and self-reflects on grounding (Self-RAG).
    """

    SYSTEM_PROMPT = (
        "You are an autonomous Enterprise Agentic RAG specialist. "
        "Your task is to answer the user query with extreme factual precision.\n\n"
        "REASONING & ACTION RULES:\n"
        "1. For each step, think about what piece of information you need and call the appropriate tool.\n"
        "2. To explore relationships or dependencies, use `query_knowledge_graph`.\n"
        "3. To retrieve policy text, specifications, or details, use `search_hybrid`.\n"
        "4. To compare two entities or policies, use `compare_entities`.\n"
        "5. Every factual statement in your final answer MUST cite the source using the exact bracket tags "
        "provided in the observations, e.g., [doc_id#chunk_id].\n"
        "6. If you have gathered enough verified information, formulate the final answer directly without calling more tools.\n"
        "7. If the documents do not contain the answer after checking, clearly state that you cannot confirm the answer based on available documentation."
    )

    def __init__(
        self,
        tool_registry: ToolRegistry,
        llm_provider: LLMProvider,
        event_bus: Optional[AgentEventBus] = None,
        max_iterations: int = 5,
        critic: Optional[Any] = None
    ):
        self.tools = tool_registry
        self.llm = llm_provider
        self.event_bus = event_bus or AgentEventBus()
        self.planner = AgentPlanner(self.llm)
        self.max_iterations = max_iterations
        self.critic = critic

    async def _emit_event(self, bus: AgentEventBus, event_type: str, **kwargs):
        await bus.emit(event_type, **kwargs)
        if bus is not self.event_bus:
            await self.event_bus.emit(event_type, **kwargs)

    async def run(
        self, 
        query: str, 
        user_roles: Optional[List[str]] = None,
        event_bus: Optional[AgentEventBus] = None
    ) -> VerifiedResponse:
        bus = event_bus or self.event_bus
        start_time = time.perf_counter()
        scratchpad = AgentScratchpad(user_query=query, max_iterations=self.max_iterations)

        # 1. Planning Phase
        subgoals = await self.planner.generate_plan(query)
        scratchpad.plan_subgoals = subgoals
        await self._emit_event(bus, "ON_PLAN", query=query, subgoals=subgoals)

        # 2. ReAct Loop
        final_answer = None
        citations = []

        if self.llm.provider_type in ["groq", "openai"]:
            final_answer, citations = await self._run_llm_react_loop(query, scratchpad, user_roles, bus=bus)
        else:
            final_answer, citations = await self._run_local_react_loop(query, scratchpad, user_roles, bus=bus)

        # Normalize unicode spaces so consoles and formatters handle text safely
        final_answer = final_answer.replace('\u202f', ' ').replace('\xa0', ' ')

        # 3. Post-Generation Self-Reflection & Security
        is_grounded, faith_score, missing = HallucinationGrader.grade_grounding(final_answer, scratchpad)
        if "cannot confirm" in final_answer.lower():
            is_grounded = False
            faith_score = 0.0

        # Reinforce with AnswerVerificationCritic if available
        if self.critic and scratchpad.gathered_passages and not final_answer.lower().startswith("i cannot confirm"):
            try:
                from src.common.models import RetrievalCandidate, Chunk
                candidates = []
                for p in scratchpad.gathered_passages:
                    c = Chunk(
                        chunk_id=p.get("chunk_id", "c_unknown"),
                        doc_id=p.get("doc_id", "doc_unknown"),
                        text=p.get("text", "")
                    )
                    candidates.append(RetrievalCandidate(chunk=c, source_strategy="hybrid", final_score=1.0))
                ver_res = self.critic.verify_answer(final_answer, candidates)
                if ver_res.claims:
                    is_grounded = is_grounded and ver_res.passed
                    faith_score = round(ver_res.faithfulness_score, 2)
            except Exception:
                pass

        sanitized_answer = SecurityManager.mask_pii(final_answer)
        total_latency = (time.perf_counter() - start_time) * 1000

        await self._emit_event(
            bus,
            "ON_ANSWER",
            answer=sanitized_answer,
            citations=citations,
            faithfulness=faith_score,
            latency_ms=round(total_latency, 2)
        )

        return VerifiedResponse(
            query=query,
            answer=sanitized_answer,
            citations=citations,
            faithfulness_score=faith_score,
            verification_passed=is_grounded,
            retries_used=max(0, scratchpad.current_iteration - 1),
            fallback_used=not is_grounded,
            latency_ms=round(total_latency, 2),
            trace_events=[s.model_dump() for s in scratchpad.steps]
        )

    async def _run_llm_react_loop(
        self, 
        query: str, 
        scratchpad: AgentScratchpad, 
        user_roles: Optional[List[str]] = None,
        bus: Optional[AgentEventBus] = None
    ) -> tuple[str, List[str]]:
        """
        Executes ReAct reasoning loop using Groq/OpenAI native Tool/Function Calling.
        """
        bus = bus or self.event_bus
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"User Question: {query}\nInitial Sub-goals: {scratchpad.plan_subgoals}"}
        ]
        tool_specs = self.tools.get_openai_tool_specs()

        for step in range(self.max_iterations):
            resp = await self.llm.chat_completion(messages, tools=tool_specs, temperature=0.1)
            
            if resp.get("status") != "success":
                # Fall back to local synthesis on API error
                return await self._run_local_react_loop(query, scratchpad, user_roles, bus=bus)

            content = resp.get("content")
            tool_calls = resp.get("tool_calls", [])

            # Check if LLM emitted thought or final answer
            if content and not tool_calls:
                # LLM determined it has sufficient evidence
                tags = re.findall(r'[\[\u3010]\s*([a-zA-Z0-9_\-]+#[a-zA-Z0-9_\-]+)\s*[\]\u3011]', content)
                extracted_citations = [f"[{t}]" for t in set(tags)]
                return content, extracted_citations

            # Process Tool Calls
            if tool_calls:
                for tc in tool_calls:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except Exception:
                        fn_args = {}

                    # Securely inject user_roles into retrieval tools if not explicitly supplied
                    if user_roles and "role_filter" not in fn_args:
                        fn_args["role_filter"] = user_roles

                    thought_text = content or f"Calling tool {fn_name} to fulfill plan."
                    await self._emit_event(bus, "ON_THOUGHT", thought=thought_text, step=step + 1)
                    await self._emit_event(bus, "ON_ACTION", tool=fn_name, args=fn_args, step=step + 1)

                    # Execute Tool
                    tool_res = await self.tools.execute_tool(fn_name, fn_args)
                    obs = tool_res.get("output", tool_res.get("error"))
                    
                    # Store passages if hybrid search
                    if fn_name == "search_hybrid" and isinstance(obs, dict):
                        for p in obs.get("passages", []):
                            scratchpad.gathered_passages.append(p)

                    # CRAG: Grade observation relevance
                    is_rel, conf, reason = RetrievalGrader.grade_observation(query, obs)
                    await self._emit_event(bus, "ON_OBSERVATION", tool=fn_name, observation=obs, is_relevant=is_rel)

                    scratchpad.add_step(
                        thought=thought_text,
                        tool_name=fn_name,
                        tool_args=fn_args,
                        observation=obs
                    )

                    # Append to LLM messages
                    messages.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [tc]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{step}"),
                        "name": fn_name,
                        "content": json.dumps(obs)
                    })

        # Max iterations reached, force synthesis from gathered scratchpad
        return self._synthesize_from_scratchpad(query, scratchpad)

    async def _run_local_react_loop(
        self, 
        query: str, 
        scratchpad: AgentScratchpad, 
        user_roles: Optional[List[str]] = None,
        bus: Optional[AgentEventBus] = None
    ) -> tuple[str, List[str]]:
        """
        Deterministic, offline ReAct policy executing tools autonomously based on the planned goals.
        """
        bus = bus or self.event_bus
        for idx, goal in enumerate(scratchpad.plan_subgoals[:self.max_iterations]):
            thought = f"Autonomous Agent Step {idx + 1}: Focusing on sub-goal '{goal}'."
            await self._emit_event(bus, "ON_THOUGHT", thought=thought, step=idx + 1)

            # Choose tool based on sub-goal semantics
            if "knowledge graph" in goal.lower() or "explore" in goal.lower():
                entities = re.findall(r'\b[A-Z][a-zA-Z0-9_\-\.]{2,}\b', goal) or re.findall(r'\b[A-Z][a-zA-Z0-9_\-\.]{2,}\b', query)
                ent = entities[0] if entities else "System"
                tool_name = "query_knowledge_graph"
                tool_args = {"entity": ent, "max_hops": 2, "role_filter": user_roles}
            elif "compare" in goal.lower():
                entities = re.findall(r'\b[A-Z][a-zA-Z0-9_\-\.]{2,}\b', query)
                ent_a = entities[0] if len(entities) > 0 else "EntityA"
                ent_b = entities[1] if len(entities) > 1 else "EntityB"
                tool_name = "compare_entities"
                tool_args = {"entity_a": ent_a, "entity_b": ent_b, "role_filter": user_roles}
            else:
                tool_name = "search_hybrid"
                tool_args = {"query": query, "top_k": 3, "role_filter": user_roles}

            await self._emit_event(bus, "ON_ACTION", tool=tool_name, args=tool_args, step=idx + 1)
            tool_res = await self.tools.execute_tool(tool_name, tool_args)
            obs = tool_res.get("output", tool_res.get("error"))

            if tool_name == "search_hybrid" and isinstance(obs, dict):
                for p in obs.get("passages", []):
                    scratchpad.gathered_passages.append(p)

            is_rel, conf, reason = RetrievalGrader.grade_observation(query, obs)
            await self._emit_event(bus, "ON_OBSERVATION", tool=tool_name, observation=obs, is_relevant=is_rel)

            scratchpad.add_step(
                thought=thought,
                tool_name=tool_name,
                tool_args=tool_args,
                observation=obs
            )

        return self._synthesize_from_scratchpad(query, scratchpad)

    def _synthesize_from_scratchpad(self, query: str, scratchpad: AgentScratchpad) -> tuple[str, List[str]]:
        if not scratchpad.gathered_passages:
            # Check if any step observations have chunks
            for s in scratchpad.steps:
                if isinstance(s.observation, dict) and s.observation.get("source_chunks"):
                    scratchpad.gathered_passages.extend(s.observation["source_chunks"])

        if not scratchpad.gathered_passages:
            return (
                "I cannot confirm an answer to this question based on the available documentation. "
                "The autonomous agent executed all planned tools but found no verified evidence.",
                []
            )

        STOP_WORDS = {"what", "when", "where", "which", "who", "whom", "how", "does", "can", "will", "should", "is", "are", "was", "were", "explain", "describe", "explore", "retrieve", "please", "tell", "the", "and", "for", "with", "this", "that", "from", "have", "has", "about", "your", "its", "all", "any"}
        q_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower())) - STOP_WORDS
        if not q_tokens:
            q_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower()))

        matched_sentences = []
        citations = []

        for p in scratchpad.gathered_passages:
            tag = p.get("citation_tag", "")
            text = p.get("text", "")
            for sent in re.split(r'(?<=[.!?])\s+', text):
                s_clean = sent.strip()
                s_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s_clean.lower())) - STOP_WORDS
                overlap = len(q_tokens.intersection(s_tokens))
                if overlap >= 1 and len(s_clean) > 20:
                    matched_sentences.append((s_clean, tag, overlap))

        if not matched_sentences:
            return (
                "I cannot confirm an answer to this question based on the available documentation. "
                "The autonomous agent executed all planned tools but found no verified evidence.",
                []
            )


        matched_sentences.sort(key=lambda x: x[2], reverse=True)

        top_sentences = matched_sentences[:3]

        answer_parts = [f"{s.rstrip('.')} {tag}." for s, tag, _ in top_sentences]
        citations = list(set([tag for _, tag, _ in top_sentences if tag]))
        final_answer = " ".join(answer_parts)

        return final_answer, citations
