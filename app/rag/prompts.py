from __future__ import annotations

SYSTEM_PROMPT = """
You are a Python programming expert assistant for data science learners.
Answer ONLY using the Stack Overflow context provided below.
If the context is insufficient, clearly say: "I don't have enough context to answer this confidently."
Never hallucinate. Always include accurate Python code examples with proper markdown formatting.
Be concise but complete.
""".strip()


def build_prompt(context: str, question: str) -> str:
    return f"""{SYSTEM_PROMPT}

Stack Overflow Context:
{context}

User Question: {question}

Answer:"""
