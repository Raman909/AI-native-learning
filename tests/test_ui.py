from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_root_serves_ui(client: AsyncClient):
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Python Q&A Assistant" in response.text


@pytest.mark.asyncio
async def test_static_assets_served(client: AsyncClient):
    response = await client.get("/static/app.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
