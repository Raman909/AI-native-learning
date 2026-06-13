from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings
from app.rag.pipeline import RAGPipeline
from app.routes.qa import router as qa_router


def build_pipeline(app: FastAPI) -> None:
    if not hasattr(app.state, "settings"):
        app.state.settings = Settings()
    if not hasattr(app.state, "rag"):
        app.state.rag = RAGPipeline(app.state.settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    build_pipeline(app)
    yield


app = FastAPI(title="Python Q&A Assistant", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(qa_router)
build_pipeline(app)


@app.get("/")
async def root() -> dict:
    return {"message": "Python Q&A Assistant", "docs": "/docs", "health": "/api/v1/health"}
