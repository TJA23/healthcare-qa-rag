import json
import re
from typing import Iterable

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, CORPUS_PATH


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current))
            tail_words = " ".join(current).split()[-overlap:] if overlap > 0 else []
            current = tail_words.copy()
            current_len = len(current)
        current.extend(words)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


def load_corpus(path=CORPUS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chunks() -> list[dict]:
    docs = load_corpus()
    out: list[dict] = []
    for doc_id, doc in enumerate(docs):
        for chunk_id, chunk in enumerate(chunk_text(doc["text"])):
            out.append(
                {
                    "id": f"{doc_id}-{chunk_id}",
                    "doc_id": doc_id,
                    "title": doc["title"],
                    "text": chunk,
                }
            )
    return out
