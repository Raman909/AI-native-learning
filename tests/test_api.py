from __future__ import annotations

import asyncio
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture(autouse=True)
def mock_llm_router() -> None:
    def fake_generate(prompt: str) -> tuple[str, str]:
        question_marker = "User Question:"
        question = prompt.split(question_marker, 1)[1].split("\n\nAnswer:", 1)[0].strip().lower() if question_marker in prompt else prompt.lower()
        if "pasta" in question:
            return "I don't have enough context to answer this confidently.", "huggingface"
        if "python not working" in question:
            return "I don't have enough context to answer this confidently.", "gemini"
        return (
            "```python\nprint('hello from the assistant')\n```",
            "gemini",
        )

    app.state.rag.llm_router.generate = fake_generate


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"


@pytest.mark.asyncio
async def test_llm_status_endpoint(client: AsyncClient):
    response = await client.get("/api/v1/llm-status")
    assert response.status_code == 200
    body = response.json()
    assert "active_provider" in body


@pytest.mark.asyncio
async def test_ask_valid_question(client: AsyncClient):
    response = await client.post(
        "/api/v1/ask",
        json={"question": "How do I reverse a list in Python?", "max_sources": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]


@pytest.mark.asyncio
async def test_ask_too_short(client: AsyncClient):
    response = await client.post("/api/v1/ask", json={"question": "hi"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ask_too_long(client: AsyncClient):
    response = await client.post("/api/v1/ask", json={"question": "a" * 600})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ask_returns_provider(client: AsyncClient):
    response = await client.post(
        "/api/v1/ask",
        json={"question": "How to handle missing values in a pandas DataFrame?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider_used"] in ["gemini", "huggingface"]


@pytest.mark.asyncio
async def test_ask_returns_sources(client: AsyncClient):
    response = await client.post(
        "/api/v1/ask",
        json={"question": "What causes RecursionError in Python and how to fix it?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sources"]


@pytest.mark.asyncio
async def test_ask_processing_time(client: AsyncClient):
    response = await client.post(
        "/api/v1/ask",
        json={"question": "How do I implement multiple inheritance in Python?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["processing_time_ms"] > 0


@pytest.mark.asyncio
async def test_concurrent_requests(client: AsyncClient):
    payloads = [
        {"question": "How do I reverse a list in Python?"},
        {"question": "How to handle missing values in a pandas DataFrame?"},
        {"question": "How does Python's GIL affect multithreading?"},
    ]

    responses = await asyncio.gather(*[client.post("/api/v1/ask", json=payload) for payload in payloads])
    assert all(response.status_code == 200 for response in responses)


@pytest.mark.asyncio
async def test_examples_endpoint(client: AsyncClient):
    response = await client.get("/api/v1/examples")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5
