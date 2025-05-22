
import os
from dotenv import load_dotenv

load_dotenv()

# ChromaDB Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Ollama Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")  # or "mistral", "codellama", etc.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Matching Parameters
MIN_CONFIDENCE_SCORE = float(os.getenv("MIN_CONFIDENCE", 0.6))