# 🏥 Healthcare Question Answering System with RAG Pipeline

> **Status:** 🚧 Codebase being rebuilt — original lost in laptop failure. Architecture, design, and results documented below.

An intelligent **medical question answering system** built on **Retrieval-Augmented Generation (RAG)**, combining **BERT** for query understanding, **FAISS** for semantic search, and **GPT** for grounded response generation — served via **FastAPI**.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![BERT](https://img.shields.io/badge/BERT-FFD21E?style=flat&logoColor=black)
![OpenAI](https://img.shields.io/badge/OpenAI_GPT-412991?style=flat&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0084FF?style=flat&logo=meta&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=flat&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Status](https://img.shields.io/badge/status-rebuilding-yellow)

---

## 🎯 Problem

Healthcare information is dense, domain-specific, and rapidly evolving. Patients and clinicians need:
- **Accurate, context-grounded answers** to medical questions
- A system that retrieves from **trusted sources** (medical articles, FAQs) instead of hallucinating
- Real-time response delivery via a clean API

LLMs alone hallucinate medical facts. **RAG fixes this** by grounding the model in a curated knowledge base.

## 🏗️ Architecture

```
┌────────────────┐    ┌──────────────┐    ┌───────────────────┐    ┌──────────┐
│  Medical Docs  │ ─► │  Cleaning &  │ ─► │  Embedding        │ ─► │  FAISS   │
│  (Articles,    │    │  Chunking    │    │  (Sentence-BERT)  │    │  Index   │
│   FAQs, PDFs)  │    └──────────────┘    └───────────────────┘    └─────┬────┘
└────────────────┘                                                        │
                                                                          │
   User Query                                                             │
       │                                                                  │
       ▼                                                                  │
┌──────────────┐    ┌──────────────────┐    ┌────────────────┐            │
│  BERT        │ ─► │  Semantic Search │ ◄──┤ Top-K Retrieve │ ◄──────────┘
│  (Intent +   │    │  (Query Embed)   │    └────────┬───────┘
│   Context)   │    └──────────────────┘             │
└──────────────┘                                     ▼
                                          ┌──────────────────┐
                                          │  GPT             │
                                          │  Context-aware   │
                                          │  Answer Gen      │
                                          └────────┬─────────┘
                                                   │
                                                   ▼
                                          ┌──────────────────┐
                                          │  FastAPI         │
                                          │  REST Endpoint   │
                                          └──────────────────┘
```

## ⚙️ Tech Stack

| Layer            | Technology                                    |
|------------------|-----------------------------------------------|
| Language         | Python 3.10+                                  |
| Query understanding | BERT (Hugging Face Transformers)           |
| Embeddings       | Sentence-BERT                                 |
| Vector store     | FAISS                                         |
| Generative model | OpenAI GPT (configurable)                     |
| API layer        | FastAPI                                       |
| Doc processing   | PyPDF2 · BeautifulSoup · LangChain text-splitter |

## 🔁 Pipeline Stages

1. **Document Ingestion** — Medical articles, FAQs, and clinical resources collected as raw documents.
2. **Cleaning & Chunking** — Strip noise; split into semantically coherent chunks (~256–512 tokens).
3. **Embedding Generation** — Encode each chunk with Sentence-BERT into dense vectors.
4. **Indexing** — Store embeddings in FAISS for sub-second nearest-neighbor search.
5. **Query Understanding** — BERT classifies user intent and extracts contextual signals.
6. **Retrieval** — Top-K most relevant chunks fetched from FAISS.
7. **Generation** — Retrieved chunks passed as context to GPT, which produces a grounded answer with citations.
8. **API Delivery** — FastAPI exposes `/ask` endpoint for real-time question answering.

## 🧠 Key Engineering Decisions

- **RAG over fine-tuning** — medical knowledge updates frequently; fine-tuning is expensive and stale.
- **BERT for intent + Sentence-BERT for retrieval** — separates *understanding* from *matching*, improving answer relevance.
- **Chunking strategy** — overlap of 50 tokens preserves context across chunk boundaries.
- **FastAPI** — async support and auto-generated OpenAPI docs make it ideal for ML serving.

## 📊 Outcomes

- Built an **intelligent healthcare QA system** delivering accurate, context-aware medical responses.
- Designed a **document ingestion pipeline** with cleaning, chunking, and embedding generation.
- Implemented **semantic search via FAISS** for relevant context retrieval before answer generation.
- Leveraged **BERT for query understanding** and **GPT for grounded response generation**.
- Exposed real-time QA via **FastAPI REST endpoints**.

## 🔮 Future Work

- Add citation/source highlighting in responses.
- Domain-adapt embeddings using **BioBERT** / **PubMedBERT**.
- Multi-turn conversational memory.
- Guardrails for harmful or out-of-scope medical advice.
- Streamlit / React UI for clinicians.

## ⚠️ Disclaimer

This is a research / educational project. Outputs are **not a substitute for professional medical advice**. Always consult a qualified healthcare professional.

## 👤 Author

**Teeja S** — Data Scientist & AI/ML Engineer
📧 teejasenthilkumar@gmail.com · 💼 [LinkedIn](https://www.linkedin.com/in/teeja-senthilkumar/) · 🌐 [Portfolio](https://datascienceportfol.io/teeja)
