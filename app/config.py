from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = DATA_DIR / "medical_faqs.json"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
CHUNK_SIZE = 350
CHUNK_OVERLAP = 60
TOP_K = 4

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
