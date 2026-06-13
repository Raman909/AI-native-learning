from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
from bs4 import BeautifulSoup

try:  # pragma: no cover - optional dependency path
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path for local imports
    LANGCHAIN_AVAILABLE = False

    @dataclass
    class Document:
        page_content: str
        metadata: dict = field(default_factory=dict)

    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size: int, chunk_overlap: int):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str) -> list[str]:
            if len(text) <= self.chunk_size:
                return [text]
            chunks: list[str] = []
            step = max(1, self.chunk_size - self.chunk_overlap)
            for start in range(0, len(text), step):
                chunks.append(text[start : start + self.chunk_size])
            return chunks

    class HuggingFaceEmbeddings:  # type: ignore[override]
        def __init__(self, model_name: str):
            self.model_name = model_name

    class _SimpleRetriever:
        def __init__(self, documents: list[Document]):
            self.documents = documents

        def invoke(self, question: str) -> list[Document]:
            tokens = {token for token in question.lower().split() if token}

            def score(document: Document) -> tuple[int, int]:
                text = document.page_content.lower()
                token_hits = sum(1 for token in tokens if token in text)
                return token_hits, -len(text)

            return sorted(self.documents, key=score, reverse=True)

    @dataclass
    class _SimpleVectorStore:
        documents: list[Document]

        @classmethod
        def from_documents(cls, documents: list[Document], embeddings: object):
            return cls(list(documents))

        @classmethod
        def from_texts(cls, texts: Iterable[str], embeddings: object, metadatas: list[dict] | None = None):
            docs = []
            metadata_items = metadatas or [{} for _ in texts]
            for index, text in enumerate(texts):
                metadata = metadata_items[index] if index < len(metadata_items) else {}
                docs.append(Document(page_content=text, metadata=metadata))
            return cls(docs)

        @classmethod
        def load_local(cls, path: str, embeddings: object, allow_dangerous_deserialization: bool = False):
            raise FileNotFoundError(path)

        def save_local(self, path: str):
            output = Path(path)
            output.mkdir(parents=True, exist_ok=True)
            (output / "seed-index.json").write_text(
                "\n".join(document.page_content for document in self.documents),
                encoding="utf-8",
            )

        def as_retriever(self, search_type: str | None = None, search_kwargs: dict | None = None):
            return _SimpleRetriever(self.documents)

    FAISS = _SimpleVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEED_DOCUMENTS = [
    Document(
        page_content=(
            "Question: How do I reverse a list in Python?\n\n"
            "Use the built-in reverse methods or slicing for a copy.\n\n"
            "Answer: You can reverse in place with list.reverse() or create a reversed copy with list[::-1].\n\n"
            "```python\nitems = [1, 2, 3]\nitems.reverse()\nprint(items)  # [3, 2, 1]\n```"
        ),
        metadata={"source": "seed", "topic": "lists"},
    ),
    Document(
        page_content=(
            "Question: How to handle missing values in a pandas DataFrame?\n\n"
            "Use isna(), fillna(), dropna(), or interpolation depending on the use case.\n\n"
            "Answer: The common approach is to inspect missingness and then choose fillna or dropna.\n\n"
            "```python\nimport pandas as pd\ndf = df.fillna(0)\n```"
        ),
        metadata={"source": "seed", "topic": "pandas"},
    ),
    Document(
        page_content=(
            "Question: What causes RecursionError in Python and how to fix it?\n\n"
            "RecursionError usually appears when recursion depth is too high or the base case is missing.\n\n"
            "Answer: Add a base case, reduce recursion depth, or rewrite the solution iteratively."
        ),
        metadata={"source": "seed", "topic": "recursion"},
    ),
    Document(
        page_content=(
            "Question: How do I implement multiple inheritance in Python?\n\n"
            "Python supports multiple inheritance through class definitions and method resolution order.\n\n"
            "Answer: Define multiple base classes and pay attention to the MRO when overriding methods."
        ),
        metadata={"source": "seed", "topic": "oop"},
    ),
    Document(
        page_content=(
            "Question: What is the fastest way to read a large CSV file in Python?\n\n"
            "Use pandas with chunking, pyarrow, or the csv module depending on memory constraints.\n\n"
            "Answer: For tabular workflows, pandas.read_csv with dtype hints and chunksize is a practical choice."
        ),
        metadata={"source": "seed", "topic": "io"},
    ),
    Document(
        page_content=(
            "Question: How does Python's GIL affect multithreading?\n\n"
            "The GIL limits simultaneous execution of Python bytecode in multiple threads.\n\n"
            "Answer: Threads still help with I/O-bound work, but CPU-bound workloads often need multiprocessing."
        ),
        metadata={"source": "seed", "topic": "gil"},
    ),
]


def clean_html(text: str) -> str:
    return BeautifulSoup(text or "", "html.parser").get_text(" ", strip=True)


def read_stackoverflow_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode dataset file: {path}")


def load_qa_pairs(questions_csv: Path, answers_csv: Path, limit: int = 50000) -> list[Document]:
    if not questions_csv.exists():
        raise FileNotFoundError(f"Questions CSV not found: {questions_csv}")
    if not answers_csv.exists():
        raise FileNotFoundError(f"Answers CSV not found: {answers_csv}")

    questions = read_stackoverflow_csv(questions_csv)
    answers = read_stackoverflow_csv(answers_csv)

    questions = questions.loc[questions["Score"] >= 5].copy()
    answers = answers.loc[answers["Score"] >= 3].copy()
    answers = answers.sort_values(["ParentId", "Score"], ascending=[True, False])
    top_answers = answers.drop_duplicates(subset=["ParentId"], keep="first")

    merged = questions.merge(
        top_answers[["ParentId", "Body", "Score"]],
        left_on="Id",
        right_on="ParentId",
        how="inner",
        suffixes=("_question", "_answer"),
    )
    merged = merged.head(limit)

    documents: list[Document] = []
    for index, row in enumerate(merged.itertuples(index=False), start=1):
        question_body = clean_html(getattr(row, "Body_question"))
        answer_body = clean_html(getattr(row, "Body_answer"))
        content = f"Question: {row.Title}\n\n{question_body}\n\nAnswer: {answer_body}"
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "question_id": int(getattr(row, "Id")),
                    "answer_score": int(getattr(row, "Score_answer")),
                    "question_score": int(getattr(row, "Score_question")),
                },
            )
        )
        if index % 5000 == 0:
            print(f"Processed {index} merged QA pairs")

    return documents


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Document] = []
    for index, document in enumerate(documents, start=1):
        split_texts = splitter.split_text(document.page_content)
        for chunk_index, chunk_text in enumerate(split_texts, start=1):
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={**document.metadata, "chunk_index": chunk_index},
                )
            )
        if index % 5000 == 0:
            print(f"Chunked {index} documents")
    return chunks


def build_faiss_index(
    questions_csv: Path,
    answers_csv: Path,
    output_dir: Path,
    embedding_model: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    limit: int = 50000,
) -> FAISS:
    merged_documents = load_qa_pairs(questions_csv, answers_csv, limit=limit)
    chunked_documents = chunk_documents(merged_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.from_documents(chunked_documents, embeddings)
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(output_dir))
    return vector_store


def load_vector_store(vector_store_path: str, embedding_model: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    path = Path(vector_store_path)
    try:
        return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return FAISS.from_documents(SEED_DOCUMENTS, embeddings)
