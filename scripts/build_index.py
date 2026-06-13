from __future__ import annotations

import argparse
from pathlib import Path

from app.config import Settings
from app.rag.ingest import build_faiss_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Stack Overflow Python QA FAISS index")
    parser.add_argument("--data-dir", default="data", help="Directory containing Questions.csv and Answers.csv")
    parser.add_argument("--output-dir", default="vector_store", help="Directory where the vector store is saved")
    parser.add_argument("--limit", type=int, default=50000, help="Maximum number of merged QA pairs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    questions_csv = data_dir / "Questions.csv"
    answers_csv = data_dir / "Answers.csv"

    print(f"Building index from {questions_csv} and {answers_csv}")
    vector_store = build_faiss_index(
        questions_csv=questions_csv,
        answers_csv=answers_csv,
        output_dir=output_dir,
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        limit=args.limit,
    )
    print(f"Saved FAISS index to {output_dir}")
    print(f"Indexed documents: {len(getattr(vector_store, 'docstore', {})) if hasattr(vector_store, 'docstore') else 'available'}")


if __name__ == "__main__":
    main()
