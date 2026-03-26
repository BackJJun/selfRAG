import logging
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=DOTENV_PATH, override=False)

LOG_LEVEL = os.getenv("SELF_RAG_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger("self_rag")
logger.info("=" * 80)
logger.info("LangGraph Self-RAG API for Python 3.11")
logger.info("=" * 80)

DEFAULT_PDF_PATH = BASE_DIR / "data" / "SPRi AI Brief_10월호_산업동향_1002_F.pdf"
DEFAULT_TEXT_PATH = BASE_DIR / "data" / "Sample_Text.txt"

raw_source_path = os.getenv("SELF_RAG_SOURCE_PATH", str(DEFAULT_PDF_PATH))
configured_source_path = Path(raw_source_path)
if not configured_source_path.is_absolute():
    configured_source_path = (BASE_DIR / configured_source_path).resolve()
FILE_PATH = configured_source_path

OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-5.4-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "").strip()
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "self-rag")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_TRACING_REQUESTED = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
LANGSMITH_TRACING = LANGSMITH_TRACING_REQUESTED and bool(LANGSMITH_API_KEY)

if LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = "true"
elif LANGSMITH_TRACING_REQUESTED:
    os.environ["LANGSMITH_TRACING"] = "false"

CHUNK_SIZE = int(os.getenv("SELF_RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("SELF_RAG_CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("SELF_RAG_TOP_K", "3"))
MAX_RETRIES = int(os.getenv("SELF_RAG_MAX_RETRIES", "2"))
STREAM_CHUNK_SIZE = int(os.getenv("SELF_RAG_STREAM_CHUNK_SIZE", "18"))

logger.info(
    "Tracing     | langsmith_enabled=%s | project=%s | endpoint=%s",
    LANGSMITH_TRACING,
    LANGSMITH_PROJECT,
    LANGSMITH_ENDPOINT,
)
if LANGSMITH_TRACING_REQUESTED and not LANGSMITH_API_KEY:
    logger.warning("Tracing     | LANGSMITH_TRACING requested but LANGSMITH_API_KEY is empty; tracing disabled")
