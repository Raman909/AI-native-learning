from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import Settings
from app.rag.pipeline import RAGPipeline
from app.routes.qa import router as qa_router

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


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
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
build_pipeline(app)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request, "index.html")
