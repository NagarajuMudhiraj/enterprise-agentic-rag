import json
import logging
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
from src.monitoring.tracer import PipelineTracer
from src.common.metrics import metrics
from src.prompts import PROMPTS_VERSION, load_prompt
from src.verification.claim_extractor import ClaimExtractor

logger = logging.getLogger(__name__)

class ReActAgentEngine:
    """
    Autonomous ReAct (Reasoning + Action) & Plan-and-Solve Agent Core.
    Dynamically plans sub-goals, invokes tools via function calling,
    evaluates observations (CRAG), and self-reflects on grounding (Self-RAG).
    """

    # Prompts live in src/prompts/*.md and are versioned via PROMPTS_VERSION.
    SYSTEM_PROMPT = load_prompt("system_agent")

    FINAL_ANSWER_PROMPT = load_prompt("final_answer")

    MAX_OBSERVATION_CHARS = 6000

    # Phrases a model uses to decline an answer. Kept in one place so the ReAct
    # loop and the post-generation verdict cannot drift apart.
    ABSTENTION_MARKERS = (
        "cannot confirm",
        "can't confirm",
        "unable to",
        "cannot provide",
        "cannot answer",
        "not available",
        "no verified evidence",
        "no evidence",
        "do not contain",
        "does not contain",
        "doesn't contain",
        "no information",
        "not mentioned",
        "not found",
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
        tracer = PipelineTracer(query_id=f"agent-{int(start_time * 1000) % 10**9}")
        scratchpad = AgentScratchpad(user_query=query, max_iterations=self.max_iterations)

        # 1. Planning Phase
        t0 = time.perf_counter()
        subgoals = await self.planner.generate_plan(query)
        tracer.record_span("planning", (time.perf_counter() - t0) * 1000, {"subgoals": len(subgoals)})
        scratchpad.plan_subgoals = subgoals
        await self._emit_event(bus, "ON_PLAN", query=query, subgoals=subgoals)

        # 2. ReAct Loop
        final_answer = None
        citations = []

        if self.llm.provider_type in ["groq", "openai"]:
            final_answer, citations = await self._run_llm_react_loop(query, scratchpad, user_roles, bus=bus)
        else:
            final_answer, citations = await self._run_local_react_loop(query, scratchpad, user_roles, bus=bus)
        tracer.record_span(
            "react_loop",
            (time.perf_counter() - t0) * 1000,
            {"steps": len(scratchpad.steps), "path": self.llm.provider_type},
        )

        # Normalize unicode spaces/hyphens/quotes so consoles, formatters and graders handle text safely
        final_answer = ClaimExtractor.normalize_text(final_answer)

        # Drop citation tags the model invented. Every bracket tag becomes a
        # clickable chip in the UI, so a fabricated tag would open a passage that
        # never supported the answer.
        citations = self._known_citations(citations, scratchpad)

        # 3. Post-Generation Self-Reflection & Security
        final_answer, is_grounded, faith_score, unsupported_claims, abstained = self._evaluate_answer(
            query, final_answer, citations, scratchpad
        )

        sanitized_answer = SecurityManager.mask_pii(final_answer)
        total_latency = (time.perf_counter() - start_time) * 1000
        tracer.record_span("verification", 0.0, {"passed": is_grounded, "faithfulness": faith_score})

        metrics.incr("rag_queries_total", {"outcome": "verified" if is_grounded else "abstained"})
        metrics.observe("rag_query_duration_seconds", total_latency / 1000.0, {"path": "agent"})
        logger.info(
            "agent_query_completed",
            extra={
                "duration_ms": round(total_latency, 2),
                "query_chars": len(query),
                "tokens_in": self.llm.total_prompt_tokens,
                "tokens_out": self.llm.total_completion_tokens,
            },
        )

        await self._emit_event(
            bus,
            "ON_ANSWER",
            answer=sanitized_answer,
            citations=citations,
            faithfulness=faith_score,
            verification_passed=is_grounded,
            abstained=abstained,
            unsupported_claims=unsupported_claims,
            prompt_version=PROMPTS_VERSION,
            latency_ms=round(total_latency, 2)
        )

        return VerifiedResponse(
            query=query,
            answer=sanitized_answer,
            citations=citations,
            faithfulness_score=faith_score,
            verification_passed=is_grounded,
            abstained=abstained,
            retries_used=max(0, scratchpad.current_iteration - 1),
            fallback_used=not is_grounded,
            latency_ms=round(total_latency, 2),
            trace_events=[s.model_dump() for s in scratchpad.steps],
            unsupported_claims=unsupported_claims,
            prompt_version=PROMPTS_VERSION,
        )

    # ------------------------------------------------------------------ Verdict
    @staticmethod
    def _known_citations(citations: List[str], scratchpad: AgentScratchpad) -> List[str]:
        """Keep only tags that came from a retrieved passage.

        With no retrieved passages there is nothing to check against, so the
        model's tags are left alone (the grader still scores the grounding).
        """
        known = {
            p.get("citation_tag") for p in scratchpad.gathered_passages if p.get("citation_tag")
        }
        if not known:
            return citations
        return [c for c in citations if c in known]

    @classmethod
    def _is_abstention(cls, text: str, citations: Optional[List[str]] = None) -> bool:
        """True when the text declines to answer rather than answering.

        A cited answer is never treated as an abstention, even if it happens to
        contain a phrase like "not found" about a detail inside the evidence.
        """
        if citations:
            return False
        lowered = (text or "").lower()
        return any(marker in lowered for marker in cls.ABSTENTION_MARKERS)

    def _evaluate_answer(
        self,
        query: str,
        answer: str,
        citations: List[str],
        scratchpad: AgentScratchpad,
    ) -> tuple[str, bool, float, List[str], bool]:
        """Scores a draft answer and decides whether the system is abstaining.

        Returns (answer, is_grounded, faithfulness, unsupported_claims, abstained).
        """
        if self._is_abstention(answer, citations):
            # Nothing was asserted, so nothing can be "unsupported". Grading the
            # refusal text itself used to surface the model's own apology as
            # evidence-less claims and report a meaningless faithfulness score.
            pointer = self._closest_evidence(query, scratchpad)
            if pointer:
                answer = f"{answer.rstrip()}\n\n{pointer}"
            return answer, False, 0.0, [], True

        is_grounded, faith_score, _ = HallucinationGrader.grade_grounding(answer, scratchpad)

        # Reinforce with AnswerVerificationCritic if available
        unsupported_claims: List[str] = []
        if self.critic and scratchpad.gathered_passages:
            try:
                from src.common.models import RetrievalCandidate, Chunk
                candidates = []
                for p in scratchpad.gathered_passages:
                    c = Chunk(
                        chunk_id=p.get("chunk_id", "c_unknown"),
                        doc_id=p.get("doc_id", "doc_unknown"),
                        text=p.get("text", "")
                    )
                    candidates.append(RetrievalCandidate(chunk=c, source_strategy="hybrid", score=1.0))
                ver_res = self.critic.verify_answer(answer, candidates)
                if ver_res.claims:
                    is_grounded = is_grounded and ver_res.passed
                    faith_score = round(ver_res.faithfulness_score, 2)
                    unsupported_claims = list(ver_res.missing_evidence_claims)
            except Exception:
                # Never silently swallow: a broken critic must be visible in logs.
                logger.exception("Critic verification failed; keeping grader score")

        return answer, is_grounded, faith_score, unsupported_claims, False

    def _closest_evidence(self, query: str, scratchpad: AgentScratchpad, limit: int = 2) -> str:
        """One line naming the passages that shared the most query terms.

        An abstention that says nothing else is a dead end. Naming the closest
        evidence turns it into a next step the user can click, while only
        quoting passages that actually share terms with the query.
        """
        scored = []
        seen = set()
        for p in scratchpad.gathered_passages:
            tag = p.get("citation_tag") or ""
            if not tag or tag in seen:
                continue
            seen.add(tag)
            relevance = RetrievalGrader.passage_relevance(query, p.get("text", ""))
            if relevance >= 0.34:
                scored.append((relevance, tag, p.get("source") or "Untitled Document"))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)
        listed = ", ".join(f"{tag} ({title})" for _, tag, title in scored[:limit])
        return f"Closest matching passages in your documents: {listed}."

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
                # API error (e.g. rate limit): answer from what was already gathered, else run offline policy
                if scratchpad.gathered_passages:
                    final = await self._llm_final_answer(query, scratchpad)
                    return final or self._synthesize_from_scratchpad(query, scratchpad)
                return await self._run_local_react_loop(query, scratchpad, user_roles, bus=bus)

            content = resp.get("content")
            reasoning = resp.get("reasoning")
            tool_calls = resp.get("tool_calls", [])

            # Check if model formatted a JSON tool call inside text instead of native function calling
            if not tool_calls and content:
                json_matches = re.findall(r'(\{[^{}]+\})', content)
                for jm in reversed(json_matches):
                    try:
                        parsed = json.loads(jm)
                        if isinstance(parsed, dict):
                            inferred_tool = None
                            if "entity" in parsed and self.tools.get_tool("query_knowledge_graph"):
                                inferred_tool = "query_knowledge_graph"
                            elif "query" in parsed and self.tools.get_tool("search_hybrid"):
                                inferred_tool = "search_hybrid"
                            elif "entity_a" in parsed and self.tools.get_tool("compare_entities"):
                                inferred_tool = "compare_entities"
                            
                            if inferred_tool:
                                tool_calls = [{
                                    "id": f"text_call_{step}",
                                    "type": "function",
                                    "function": {
                                        "name": inferred_tool,
                                        "arguments": jm
                                    }
                                }]
                                content = content.replace(jm, "").strip()
                                break
                    except Exception:
                        pass

            # Check if LLM emitted thought or final answer
            if content and not tool_calls:
                tags = re.findall(r'[\[\u3010]\s*([a-zA-Z0-9_\-]+#[a-zA-Z0-9_\-]+)\s*[\]\u3011]', content)
                extracted_citations = list(dict.fromkeys([f"[{t}]" for t in tags]))

                # 1. If it cited evidence, it is a grounded final answer
                if extracted_citations:
                    return content, extracted_citations

                # 2. If it is an explicit abstention answer
                if self._is_abstention(content, extracted_citations) or step == self.max_iterations - 1:
                    return content, extracted_citations

            if not tool_calls:
                # Neither a cited answer nor a tool call: nudge the model instead of repeating the same request
                if content:
                    messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        "Answer the question now using the evidence gathered, citing each fact with the exact "
                        "bracket tags from the tool results, or call search_hybrid if you still need evidence."
                    )
                })
                continue

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

                    thought_text = content or reasoning or f"Calling tool {fn_name} to fulfill plan."
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
                        "content": content or "",
                        "tool_calls": [tc]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"call_{step}"),
                        "name": fn_name,
                        "content": self._compact_observation(obs)
                    })

        # Max iterations reached: synthesize a final answer from gathered evidence
        if scratchpad.gathered_passages:
            final = await self._llm_final_answer(query, scratchpad)
            if final:
                return final
        return self._synthesize_from_scratchpad(query, scratchpad)

    def _compact_observation(self, obs: Any) -> str:
        """
        Serializes tool output for the LLM with only the fields it needs, keeping requests small.
        """
        if isinstance(obs, dict) and isinstance(obs.get("passages"), list):
            obs = {
                "results": [
                    {"tag": p.get("citation_tag"), "source": p.get("source", ""), "text": p.get("text", "")}
                    for p in obs["passages"]
                ]
            }
        text = json.dumps(obs, ensure_ascii=False, default=str)
        if len(text) > self.MAX_OBSERVATION_CHARS:
            text = text[:self.MAX_OBSERVATION_CHARS] + "...(truncated)"
        return text

    async def _llm_final_answer(self, query: str, scratchpad: AgentScratchpad) -> Optional[tuple[str, List[str]]]:
        """
        Asks the LLM (without tools) to write a cited answer from the evidence gathered so far.
        """
        if self.llm.provider_type not in ["groq", "openai"]:
            return None

        seen = set()
        evidence_blocks = []
        for p in scratchpad.gathered_passages:
            tag = p.get("citation_tag", "")
            if not tag or tag in seen:
                continue
            seen.add(tag)
            evidence_blocks.append(f"{tag}\n{p.get('text', '')}")
            if len(evidence_blocks) >= 8:
                break
        if not evidence_blocks:
            return None

        messages = [
            {"role": "system", "content": self.FINAL_ANSWER_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nEvidence passages:\n\n" + "\n\n".join(evidence_blocks)}
        ]
        resp = await self.llm.chat_completion(messages, temperature=0.1, max_tokens=1200)
        content = resp.get("content") if resp.get("status") == "success" else None
        if not content:
            return None

        tags = re.findall(r'[\[\u3010]\s*([a-zA-Z0-9_\-]+#[a-zA-Z0-9_\-]+)\s*[\]\u3011]', content)
        return content, list(dict.fromkeys([f"[{t}]" for t in tags]))

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
        citations = list(dict.fromkeys([tag for _, tag, _ in top_sentences if tag]))
        final_answer = " ".join(answer_parts)

        return final_answer, citations
