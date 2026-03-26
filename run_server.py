import logging
import os

from app.main import serve_web


HOST = os.getenv("SELF_RAG_HOST", "127.0.0.1")
PORT = int(os.getenv("SELF_RAG_PORT", "8000"))
logger = logging.getLogger("self_rag.runner")


def main() -> None:
    logger.info("Runner start | host=%s | port=%d", HOST, PORT)
    serve_web(host=HOST, port=PORT)


if __name__ == "__main__":
    main()
