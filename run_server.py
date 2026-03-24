import os
import logging
import threading
import time
import webbrowser

from langgraph_selfRAG import serve_web


HOST = os.getenv("SELF_RAG_HOST", "127.0.0.1")
PORT = int(os.getenv("SELF_RAG_PORT", "8000"))
logger = logging.getLogger("self_rag.runner")


def open_browser() -> None:
    """
    서버가 뜰 시간을 조금 준 뒤 기본 브라우저로 화면을 연다.
    """
    time.sleep(1.0)
    logger.info("Browser open | url=http://%s:%d", HOST, PORT)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    logger.info("Runner start | host=%s | port=%d", HOST, PORT)
    threading.Thread(target=open_browser, daemon=True).start()
    serve_web(host=HOST, port=PORT)
