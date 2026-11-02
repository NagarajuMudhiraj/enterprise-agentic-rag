"""Golden-set retrieval evaluation.

Unlike the contract tests, this measures *quality*: does the right passage
surface for a known question? It runs against the real retriever stack (hybrid,
graph, RRF, rerank) with no LLM, so results are deterministic and fast.

Recall is asserted against a floor rather than exact equality, so a single
borderline case does not fail the build while the aggregate still guards against
regressions. Access-control cases are asserted absolutely — a leak is a defect,
not a metric.
"""

import json
from pathlib import Path

import pytest

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "retrieval_golden.json").read_text(encoding="utf-8")
)
RECALL_FLOOR = 0.85


@pytest.fixture()
def golden_pipeline(fresh_pipeline):
    for document in GOLDEN["documents"]:
        fresh_pipeline.ingest_document(
            raw_text=document["text"],
            title=document["title"],
            allowed_roles=document["allowed_roles"],
        )
    return fresh_pipeline


async def _retrieve(pipeline, question: str, roles) -> list:
    plan = pipeline.router.route_query(question)
    return await pipeline._execute_retrieval_layer(question, plan, roles)


def _surface(candidates) -> list:
    return [
        (c.chunk.metadata.get("title", ""), (c.chunk.text or "") + " " + (c.parent_text or ""))
        for c in candidates
    ]


@pytest.mark.asyncio
async def test_golden_set_recall(golden_pipeline, capsys):
    hits = 0
    misses = []

    for case in GOLDEN["cases"]:
        candidates = await _retrieve(golden_pipeline, case["question"], case["roles"])
        surfaced = _surface(candidates)
        matched = any(
            title == case["expect_title"]
            and any(marker.lower() in blob.lower() for marker in case["expect_any"])
            for title, blob in surfaced
        )
        if matched:
            hits += 1
        else:
            misses.append((case["id"], [title for title, _ in surfaced]))

    recall = hits / len(GOLDEN["cases"])
    with capsys.disabled():
        print(f"\ngolden recall@{len(GOLDEN['cases'])}: {recall:.0%} ({hits}/{len(GOLDEN['cases'])})")
        for case_id, titles in misses:
            print(f"  miss {case_id}: surfaced {titles}")

    assert recall >= RECALL_FLOOR, f"recall {recall:.0%} below floor; misses: {misses}"


@pytest.mark.asyncio
async def test_golden_set_never_leaks_restricted_documents(golden_pipeline):
    for case in GOLDEN["isolation_cases"]:
        candidates = await _retrieve(golden_pipeline, case["question"], case["roles"])
        titles = {title for title, _ in _surface(candidates)}
        assert case["forbidden_title"] not in titles, (
            f"{case['id']}: role {case['roles']} retrieved {case['forbidden_title']}"
        )


@pytest.mark.asyncio
async def test_restricted_document_is_reachable_with_the_right_role(golden_pipeline):
    case = next(c for c in GOLDEN["cases"] if c["roles"] == ["executive"])
    candidates = await _retrieve(golden_pipeline, case["question"], ["executive"])
    assert any(title == case["expect_title"] for title, _ in _surface(candidates))


@pytest.mark.asyncio
async def test_answerable_question_passes_verification(golden_pipeline):
    """End-to-end contract on the deterministic path: answer, cite, verify."""
    response = await golden_pipeline.query(
        "When do the full backups run?", user_roles=["public"]
    )

    assert response.citations, "expected the backup schedule to be cited"
    assert response.verification_passed is True
    assert "02:00" in response.answer
    assert response.prompt_version == GOLDEN["version"]
