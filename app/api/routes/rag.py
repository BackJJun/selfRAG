from fastapi import APIRouter, HTTPException

from app.core.config import logger
from app.schemas.rag import CRAGResponse, HybridRAGResponse, SelfRAGRequest, SelfRAGResponse
from app.services.crag import result_to_payload as crag_result_to_payload
from app.services.crag import run_crag
from app.services.hybrid_rag import result_to_payload as hybrid_result_to_payload
from app.services.hybrid_rag import run_hybrid_rag
from app.services.self_rag import result_to_payload as self_rag_result_to_payload
from app.services.self_rag import run_self_rag


router = APIRouter(prefix="/api/v1", tags=["rag"])


# 요청 질문이 비어 있지 않은지 확인하고 양끝 공백을 제거한 값을 반환한다.
def normalize_question(question: str) -> str:
    normalized = question.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="question must not be empty")
    return normalized


# Hybrid-RAG 실행 결과를 응답 모델로 변환하고 공통 메타 정보를 채운다.
def build_hybrid_response(payload: SelfRAGRequest, question: str) -> HybridRAGResponse:
    result = run_hybrid_rag(question, payload.chat_history)
    response_payload = hybrid_result_to_payload(result)
    if not payload.include_trace:
        response_payload["trace"] = []
    response_payload["meta"] = {
        "history_turns": len(payload.chat_history),
        "trace_included": payload.include_trace,
        "pipeline": "hybrid_rag",
    }
    return HybridRAGResponse.model_validate(response_payload)


# 기존 호환 경로에서 요청된 pipeline에 따라 Self-RAG, CRAG, Hybrid-RAG 중 하나를 실행한다.
@router.post("/self-rag", response_model=SelfRAGResponse | CRAGResponse | HybridRAGResponse)
async def run_rag_endpoint(payload: SelfRAGRequest) -> SelfRAGResponse | CRAGResponse | HybridRAGResponse:
    question = normalize_question(payload.question)

    logger.info(
        "HTTP POST    | path=/api/v1/self-rag | pipeline=%s | question=%r | history_turns=%d",
        payload.pipeline,
        question,
        len(payload.chat_history),
    )

    if payload.pipeline == "hybrid_rag":
        return build_hybrid_response(payload, question)

    if payload.pipeline == "crag":
        result = run_crag(question, payload.chat_history)
        response_payload = crag_result_to_payload(result)
        if not payload.include_trace:
            response_payload["trace"] = []
        response_payload["meta"] = {
            "history_turns": len(payload.chat_history),
            "trace_included": payload.include_trace,
            "pipeline": payload.pipeline,
        }
        return CRAGResponse.model_validate(response_payload)

    result = run_self_rag(question, payload.chat_history)
    response_payload = self_rag_result_to_payload(result)
    if not payload.include_trace:
        response_payload["trace"] = []
    response_payload["meta"] = {
        "history_turns": len(payload.chat_history),
        "trace_included": payload.include_trace,
        "pipeline": payload.pipeline,
    }
    return SelfRAGResponse.model_validate(response_payload)


# 운영용 /think 경로는 항상 Hybrid-RAG를 사용하도록 고정한다.
@router.post("/think", response_model=HybridRAGResponse)
async def run_think_endpoint(payload: SelfRAGRequest) -> HybridRAGResponse:
    question = normalize_question(payload.question)

    logger.info(
        "HTTP POST    | path=/api/v1/think | pipeline=hybrid_rag | question=%r | history_turns=%d",
        question,
        len(payload.chat_history),
    )

    return build_hybrid_response(payload, question)
