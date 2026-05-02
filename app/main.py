from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import TOP_K
from app.rag import answer

app = FastAPI(
    title="Healthcare QA · RAG",
    description="Retrieval-Augmented Generation system answering medical questions grounded in a curated knowledge base.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What are the symptoms of type 2 diabetes?"])
    top_k: int = Field(TOP_K, ge=1, le=10)


class Source(BaseModel):
    title: str
    chunk_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


@app.get("/")
def root():
    return {
        "name": "Healthcare QA · RAG",
        "endpoints": ["/ask (POST)", "/health (GET)", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = answer(req.question, req.top_k)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            Source(title=h.title, chunk_id=h.chunk_id, score=h.score, text=h.text)
            for h in result.sources
        ],
    )
