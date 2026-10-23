import pytest
from src.pipeline import EnterpriseRAGPipeline
from src.agent.tools.registry import AgentTool, ToolRegistry
from src.agent.reflection.graders import RetrievalGrader, HallucinationGrader

SAMPLE_DATA = """# Payment Microservice Architecture

## Dependencies
PaymentGateway connects to StripeAPI for credit card processing.
PaymentGateway requires VaultService for tokenized master keys.
PaymentGateway depends on PostgreSQL for ledger transaction logs.

## SLA & Operational Rules
PaymentGateway mandates PCI-DSS level 1 compliance.
Transaction logs are archived for 7 years in secure cloud vaults.
"""

@pytest.mark.asyncio
async def test_agent_tool_registry():
    registry = ToolRegistry()
    
    async def sample_calc(a: int, b: int) -> int:
        return a + b

    tool = AgentTool(
        name="calculator",
        description="Adds two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        },
        func=sample_calc
    )
    registry.register(tool)
    assert len(registry.list_tools()) == 1
    
    # Test with extraneous arguments (should filter safely and not raise TypeError)
    res = await registry.execute_tool("calculator", {"a": 10, "b": 20, "extra_arg": "ignore_me"})
    assert res["status"] == "success"
    assert res["output"] == 30

    # Test synchronous function via thread
    def sync_multiply(x: int, y: int) -> int:
        return x * y

    tool_sync = AgentTool(
        name="multiplier",
        description="Multiplies two numbers",
        parameters={"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}},
        func=sync_multiply
    )
    registry.register(tool_sync)
    res_sync = await registry.execute_tool("multiplier", {"x": 6, "y": 7, "unknown": True})
    assert res_sync["status"] == "success"
    assert res_sync["output"] == 42

@pytest.mark.asyncio
async def test_agentic_rag_execution():
    pipeline = EnterpriseRAGPipeline()
    pipeline.ingest_document(SAMPLE_DATA, title="Payment Architecture")

    # 1. Test multi-hop relational inquiry
    res = await pipeline.query("What does PaymentGateway depend on?")
    assert res.verification_passed is True
    assert "PostgreSQL" in res.answer or "VaultService" in res.answer or "StripeAPI" in res.answer
    assert len(res.citations) > 0

    # 2. Test CRAG grader
    is_rel, score, reason = RetrievalGrader.grade_observation("PostgreSQL ledger", "PaymentGateway depends on PostgreSQL for ledger transaction logs.")
    assert is_rel is True
    assert score > 0.5

    # 3. Test abstention on unanswerable query
    res_abstain = await pipeline.query("What is the recipe for vanilla ice cream?")
    assert res_abstain.fallback_used is True
    ans_lower = res_abstain.answer.lower()
    assert any(phrase in ans_lower for phrase in [
        "cannot confirm", "unable", "cannot provide", "not available",
        "no verified evidence", "do not contain", "does not contain",
        "no information", "not mentioned", "not found"
    ])
