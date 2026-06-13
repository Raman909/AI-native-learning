from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import AskRequest, AskResponse, HealthResponse, SourceDocument

router = APIRouter(prefix="/api/v1", tags=["qa"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    pipeline = request.app.state.rag
    return HealthResponse(
        status="healthy",
        llm_status=pipeline.llm_router.get_status(),
        vector_store="loaded",
    )


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, payload: AskRequest) -> AskResponse:
    pipeline = request.app.state.rag
    start_time = time.perf_counter()
    try:
        result = await run_in_threadpool(pipeline.ask, payload.question, payload.max_sources)
    except Exception as exception:
        message = str(exception)
        if "Both LLM providers failed" in message:
            raise HTTPException(status_code=503, detail="LLM services unavailable, try again later") from exception
        raise HTTPException(status_code=503, detail="LLM services unavailable, try again later") from exception

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return AskResponse(
        question=payload.question,
        answer=result["answer"],
        sources=[SourceDocument(**source) for source in result["sources"]],
        model=result["model"],
        provider_used=result["provider_used"],
        processing_time_ms=elapsed_ms,
    )


@router.get("/examples")
async def examples() -> list[str]:
    return [
        "How do I reverse a list in Python?",
        "How to handle missing values in a pandas DataFrame?",
        "What causes RecursionError in Python and how do I fix it?",
        "How do I implement multiple inheritance in Python?",
        "What is the fastest way to read a large CSV file in Python?",
    ]


@router.get("/llm-status")
async def llm_status(request: Request) -> dict:
    return request.app.state.rag.llm_router.get_status()
