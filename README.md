# 🏥 Healthcare Question Answering System with RAG Pipeline

> ✅ **Status:** Working MVP — runs locally end-to-end. FastAPI server, FAISS vector store, Sentence-BERT embeddings, and Groq/OpenAI-ready LLM layer.

A medical question-answering system built on **Retrieval-Augmented Generation (RAG)**. Combines **Sentence-BERT** for semantic search, **FAISS** for fast vector retrieval, and a configurable **LLM** (Groq / OpenAI) for grounded, citation-backed answers — served via **FastAPI**.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0084FF?logo=meta&logoColor=white)
![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-FFD21E?logoColor=black)
![Groq](https://img.shields.io/badge/Groq-F55036?logoColor=white)
![Status](https://img.shields.io/badge/status-MVP_working-brightgreen)

---

## ✨ What it does

Ask a medical question → the system embeds it, retrieves the most relevant chunks from a curated knowledge base via FAISS, then asks an LLM to generate a grounded answer that **cites its sources**.

```
You: "What are the symptoms of type 2 diabetes?"

System:
  → embeds the question
  → retrieves top-K chunks from FAISS  (top hit: score 0.81)
  → LLM generates a grounded answer with [Source: ...] citations
  → returns answer + raw retrieved passages for transparency
```

## 🏗️ Architecture

```
┌────────────────┐    ┌──────────────┐    ┌────────────────────┐    ┌───────────┐
│  Medical FAQs  │ ─► │  Chunking    │ ─► │  Sentence-BERT     │ ─► │  FAISS    │
│  (JSON corpus) │    │  (overlap)   │    │  Embeddings (384d) │    │  Index    │
└────────────────┘    └──────────────┘    └────────────────────┘    └─────┬─────┘
                                                                          │
   User Query                                                             │
       │                                                                  │
       ▼                                                                  │
┌──────────────┐    ┌──────────────────┐    ┌────────────────┐            │
│  Embed Query │ ─► │  Top-K Search    │ ◄──┤ FAISS Retrieve │ ◄──────────┘
└──────────────┘    └────────┬─────────┘    └────────────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │  LLM (Groq/OpenAI)  │
                   │  Grounded answer    │
                   │  + source citations │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │  FastAPI /ask       │
                   │  → JSON response    │
                   └─────────────────────┘
```

## 📂 Project Structure

```
healthcare-qa-rag/
├── app/
│   ├── config.py         # Paths, model names, hyperparameters
│   ├── chunking.py       # Sentence-aware chunking with overlap
│   ├── embeddings.py     # Sentence-BERT encoder
│   ├── vectorstore.py    # FAISS wrapper (build/save/load/search)
│   ├── llm.py            # Groq client + grounded prompt
│   ├── rag.py            # Orchestrates: embed → retrieve → generate
│   └── main.py           # FastAPI app (/ask, /health, /docs)
├── scripts/
│   └── build_index.py    # CLI: build FAISS index from corpus
├── data/
│   ├── medical_faqs.json # Curated medical knowledge base
│   └── index/            # Generated FAISS index + chunks
├── requirements.txt
└── .env.example
```

## ⚙️ Tech Stack

| Layer            | Choice                              | Why |
|------------------|-------------------------------------|------|
| Embeddings       | `all-MiniLM-L6-v2` (384-dim)        | Fast, small, strong baseline |
| Vector Store     | FAISS `IndexFlatIP`                 | Cosine sim via inner product on normalized vectors |
| LLM              | Groq (LLaMA 3.3 70B)                | Free tier, blazing fast inference |
| API Server       | FastAPI + Uvicorn                   | Async, auto-docs, production-ready |
| Chunking         | Sentence-aware with 60-word overlap | Preserves context across boundaries |

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/TJA23/healthcare-qa-rag.git
cd healthcare-qa-rag
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. (Optional) Add Groq API key for full LLM answers

Get a free key at https://console.groq.com → copy `.env.example` to `.env` → paste your key.
Without a key, the system still works and returns the top retrieved passages.

```bash
cp .env.example .env
# edit .env: GROQ_API_KEY=gsk_...
```

### 3. Build the index

```bash
python -m scripts.build_index
# → 15 chunks, 384-dim vectors, saved to data/index/
```

### 4. Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

### 5. Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the warning signs of a stroke?", "top_k": 3}'
```

Sample response:

```json
{
  "question": "What are the warning signs of a stroke?",
  "answer": "The warning signs of a stroke can be remembered using the FAST acronym...
             [Source: What are the warning signs of a stroke?]
             ...This is general information, not medical advice.",
  "sources": [
    {"title": "What are the warning signs of a stroke?", "score": 0.81, "text": "..."},
    {"title": "What causes migraines?", "score": 0.31, "text": "..."}
  ]
}
```

## 🧠 Key Design Decisions

- **Normalized embeddings + Inner Product** — equivalent to cosine similarity but faster, works directly with FAISS `IndexFlatIP`.
- **Sentence-aware chunking** — splits on sentence boundaries with overlap, preserving semantic continuity.
- **LLM-optional retrieval** — system gracefully degrades to returning raw retrieved passages if no LLM key is set, so the retrieval layer is independently testable.
- **Grounded prompt with citation requirement** — the LLM is explicitly instructed to use only the provided context and cite sources, reducing hallucination risk.
- **Disclaimer enforced at prompt level** — every answer ends with the medical-advice disclaimer.

## 📊 Sample Retrieval Results

| Query                                    | Top match (score) |
|------------------------------------------|-------------------|
| "What are the symptoms of type 2 diabetes?" | 0.81 ✅ correct FAQ |
| "What are the warning signs of a stroke?"   | 0.63 ✅ correct FAQ |
| "How does the flu vaccine work?"            | retrieves influenza vaccine FAQ |

## 🔮 Roadmap

- [ ] Domain-adapt embeddings with **BioBERT** / **PubMedBERT**
- [ ] Add **citation highlighting** in API responses
- [ ] Cross-encoder **re-ranking** of top-K
- [ ] **Multi-turn** conversation memory
- [ ] **Streamlit** UI for interactive use
- [ ] **Evaluation harness** (faithfulness, answer relevance)
- [ ] **Guardrails** for harmful / out-of-scope queries
- [ ] **Dockerize** for deployment

## ⚠️ Disclaimer

This is a **research and educational project**. Outputs are **NOT a substitute for professional medical advice**. Always consult a qualified healthcare professional for medical decisions.

## 👤 Author

**Teeja S** — Data Scientist & AI/ML Engineer
📧 teejasenthilkumar@gmail.com · 💼 [LinkedIn](https://www.linkedin.com/in/teeja-senthilkumar/) · 🌐 [Portfolio](https://datascienceportfol.io/teeja)
