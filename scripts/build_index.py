"""Build FAISS index from the medical FAQ corpus.

Run from project root:
    python -m scripts.build_index
"""

from app.chunking import build_chunks
from app.embeddings import embed
from app.vectorstore import VectorStore


def main() -> None:
    print("Loading & chunking corpus...")
    chunks = build_chunks()
    print(f"  → {len(chunks)} chunks")

    print("Generating embeddings...")
    vectors = embed([c["text"] for c in chunks])
    print(f"  → vectors shape: {vectors.shape}")

    print("Building FAISS index...")
    store = VectorStore()
    store.build(chunks, vectors)
    store.save()
    print("Index saved. Done.")


if __name__ == "__main__":
    main()
