from __future__ import annotations

from app.config import Settings
from app.rag.ingest import load_vector_store
from app.rag.llm_router import LLMRouter
from app.rag.prompts import build_prompt


class RAGPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vectorstore = load_vector_store(settings.vector_store_path, settings.embedding_model)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20},
        )
        self.llm_router = LLMRouter(
            gemini_api_key=settings.gemini_api_key,
            hf_api_key=settings.hf_api_key,
            gemini_model=settings.gemini_model,
            hf_model=settings.hf_model,
        )

    def ask(self, question: str, max_sources: int = 3) -> dict:
        documents = self.retriever.invoke(question)
        if not isinstance(documents, list):
            documents = list(documents)
        selected_docs = documents[:max_sources]
        context = "\n\n".join(document.page_content for document in selected_docs)
        prompt = build_prompt(context, question)
        answer, provider_used = self.llm_router.generate(prompt)
        return {
            "answer": answer,
            "provider_used": provider_used,
            "sources": [{"content": document.page_content[:300]} for document in selected_docs],
            "model": provider_used,
        }
