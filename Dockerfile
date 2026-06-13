FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    langchain==0.2.5 \
    langchain-community==0.2.5 \
    langchain-huggingface==0.0.3 \
    langchain-core==0.2.43 \
    langchain-text-splitters==0.2.4 \
    faiss-cpu==1.13.2 \
    google-generativeai==0.7.2 \
    pydantic==2.12.5 \
    pydantic-settings==2.14.1 \
    pandas==2.3.3 \
    beautifulsoup4==4.12.3 \
    requests==2.32.3 \
    httpx==0.27.0 \
    python-dotenv==1.2.2 \
    transformers==4.40.0 \
    tokenizers==0.19.1 \
    sentence-transformers==2.7.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu

COPY . .
EXPOSE 10000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]