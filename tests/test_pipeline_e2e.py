import pytest
from src.pipeline import EnterpriseRAGPipeline

SAMPLE_POLICY = """# Corporate Security Policy

## Multi-Factor Authentication
All employees must configure HardwareKey or AuthenticatorApp for accessing corporate VPN.
Session expiration is set to 8 hours.

## Encryption Standards
All customer databases must use AES-256 encryption at rest.
TLS-1.3 is mandatory for external communications.
"""

RESTRICTED_DOC = """# Confidential Executive Salaries

Executive bonus pool for Q4 is capped at 15 percent.
"""

@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    pipeline = EnterpriseRAGPipeline()

    # 1. Ingest Public Policy
    pipeline.ingest_document(SAMPLE_POLICY, title="Security Policy", allowed_roles=["public", "employee"])

    # 2. Ingest Restricted Document
    pipeline.ingest_document(RESTRICTED_DOC, title="Executive Compensation", allowed_roles=["executive"])

    # 3. Query Public Information
    res = await pipeline.query("What encryption standard is required for customer databases?", user_roles=["public"])
    assert res.verification_passed is True
    assert "AES-256" in res.answer
    assert len(res.citations) > 0
    assert res.faithfulness_score >= 0.80

    # 4. ACL Security Test: Public user cannot access restricted document
    unauthorized_res = await pipeline.query("What is the executive bonus pool?", user_roles=["public"])
    assert unauthorized_res.fallback_used is True
    assert "cannot confirm" in unauthorized_res.answer.lower()

    # 5. Abstention Test: Unanswerable query
    abstain_res = await pipeline.query("What is the rocket launch schedule for Mars?", user_roles=["public"])
    assert abstain_res.fallback_used is True
    assert "cannot confirm" in abstain_res.answer.lower()
