from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4.1-mini"

CHUNK_SIZE = 180
CHUNK_OVERLAP = 40
TOP_K = 4

DOCUMENTS_CSV = PROCESSED_DIR / "documents.csv"
CHUNKS_CSV = PROCESSED_DIR / "chunks.csv"
EMBEDDINGS_NPY = PROCESSED_DIR / "chunk_embeddings.npy"
EMBEDDINGS_META_CSV = PROCESSED_DIR / "chunk_embeddings_meta.csv"
FAISS_INDEX_PATH = PROCESSED_DIR / "faiss.index"