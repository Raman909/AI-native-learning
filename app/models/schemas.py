from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=500)
    max_sources: int = Field(default=3, ge=1, le=10)


class SourceDocument(BaseModel):
    content: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument]
    model: str
    provider_used: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    llm_status: dict
    vector_store: str
