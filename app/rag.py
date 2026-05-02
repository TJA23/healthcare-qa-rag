from dataclasses import dataclass
from functools import lru_cache

from app.config import TOP_K
from app.embeddings import embed
from app.llm import generate_answer
from app.vectorstore import Hit, VectorStore


@dataclass
class RagResponse:
    question: str
    answer: str
    sources: list[Hit]


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    store = VectorStore()
    store.load()
    return store


def answer(question: str, top_k: int = TOP_K) -> RagResponse:
    store = get_store()
    query_vec = embed([question])
    hits = store.search(query_vec, top_k)
    text = generate_answer(question, hits)
    return RagResponse(question=question, answer=text, sources=hits)
