import uvicorn
from fastapi import FastAPI

from app.api.routes import chat_router
from app.core.config import FILE_PATH, MAX_RETRIES, logger


app = FastAPI(
    title="Self-RAG API",
    version="1.0.0",
    description="Stateless Self-RAG API for upstream services.",
)
app.include_router(chat_router)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


def serve_web(host: str = "127.0.0.1", port: int = 8000) -> None:
    logger.info(
        "Server init | host=%s | port=%d | source_path=%s | max_retries=%d",
        host,
        port,
        FILE_PATH,
        MAX_RETRIES,
    )
    logger.info("Server ready | url=http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
