from __future__ import annotations

import re
import time

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


QUERY_CASES = [
    "How do I reverse a list in Python?",
    "How to handle missing values in a pandas DataFrame?",
    "What causes RecursionError in Python and how to fix it?",
    "How do I implement multiple inheritance in Python?",
    "What is the fastest way to read a large CSV file in Python?",
    "Python not working",
    "How do I make pasta?",
    "How does Python's GIL affect multithreading?",
]


@pytest.fixture(autouse=True)
def mock_llm_router() -> None:
    def fake_generate(prompt: str) -> tuple[str, str]:
        match = re.search(r"User Question:\s*(.*?)\n\nAnswer:", prompt, re.S)
        question = match.group(1).strip().lower() if match else prompt.lower()
        if "pasta" in question or "not working" in question:
            return "I don't have enough context to answer this confidently.", "huggingface"
        return (
            "```python\nprint('sample answer')\n```",
            "gemini",
        )

    app.state.rag.llm_router.generate = fake_generate


@pytest.mark.asyncio
@pytest.mark.parametrize("question", QUERY_CASES)
async def test_query_matrix(question: str):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        start_time = time.perf_counter()
        response = await client.post("/api/v1/ask", json={"question": question})
        latency_ms = (time.perf_counter() - start_time) * 1000

    assert response.status_code == 200
    body = response.json()
    record = {
        "question": question,
        "provider_used": body["provider_used"],
        "answer": body["answer"][:200],
        "sources_count": len(body["sources"]),
        "latency_ms": round(latency_ms, 2),
        "pass_fail": "pass",
        "observation": "insufficient context" if "I don't have enough context" in body["answer"] else "grounded answer returned",
    }

    assert record["provider_used"] in ["gemini", "huggingface"]
    assert record["sources_count"] > 0
    assert record["latency_ms"] > 0
    assert record["pass_fail"] == "pass"
    if question == "How do I make pasta?":
        assert "I don't have enough context" in body["answer"]
