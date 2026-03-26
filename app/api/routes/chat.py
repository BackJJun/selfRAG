from fastapi import APIRouter, HTTPException

from app.core.config import logger
from app.schemas.chat import SelfRAGRequest, SelfRAGResponse
from app.services.self_rag import result_to_payload, run_self_rag


router = APIRouter(prefix="/api/v1", tags=["self-rag"])


@router.post("/self-rag", response_model=SelfRAGResponse)
async def run_self_rag_endpoint(payload: SelfRAGRequest) -> SelfRAGResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    logger.info(
        "HTTP POST    | path=/api/v1/self-rag | question=%r | history_turns=%d",
        question,
        len(payload.chat_history),
    )

    result = run_self_rag(question, payload.chat_history)
    response_payload = result_to_payload(result)
    if not payload.include_trace:
        response_payload["trace"] = []

    response_payload["meta"] = {
        "history_turns": len(payload.chat_history),
        "trace_included": payload.include_trace,
    }
    return SelfRAGResponse.model_validate(response_payload)
