import asyncio

import httpx
import openai
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


# 파이프라인 실행 중 발생하는 예외를 종류별로 구분해 적절한 HTTP 에러로 변환한다.
def _handle_pipeline_error(exc: Exception, pipeline: str) -> None:
    if isinstance(exc, openai.RateLimitError):
        logger.warning("Pipeline error | pipeline=%s | type=rate_limit | %s", pipeline, exc)
        raise HTTPException(status_code=429, detail="OpenAI API rate limit exceeded. 잠시 후 다시 시도해주세요.")
    if isinstance(exc, openai.AuthenticationError):
        logger.error("Pipeline error | pipeline=%s | type=auth | %s", pipeline, exc)
        raise HTTPException(status_code=502, detail="OpenAI 인증 오류가 발생했습니다.")
    if isinstance(exc, openai.APIError):
        logger.error("Pipeline error | pipeline=%s | type=openai_api | %s", pipeline, exc)
        raise HTTPException(status_code=502, detail=f"OpenAI API 오류: {exc.message}")
    if isinstance(exc, httpx.TimeoutException):
        logger.error("Pipeline error | pipeline=%s | type=timeout | %s", pipeline, exc)
        raise HTTPException(status_code=504, detail="외부 API 응답 시간이 초과되었습니다.")
    if isinstance(exc, httpx.HTTPError):
        logger.error("Pipeline error | pipeline=%s | type=http | %s", pipeline, exc)
        raise HTTPException(status_code=502, detail="외부 서비스와의 통신에 실패했습니다.")
    logger.exception("Pipeline error | pipeline=%s | type=unexpected", pipeline)
    raise HTTPException(status_code=500, detail="파이프라인 실행 중 예기치 않은 오류가 발생했습니다.")


# Hybrid-RAG를 스레드 풀에서 비동기로 실행하고 응답 모델로 변환한다.
async def _build_hybrid_response(payload: SelfRAGRequest, question: str) -> HybridRAGResponse:
    try:
        result = await asyncio.to_thread(run_hybrid_rag, question, payload.chat_history)
    except Exception as exc:
        _handle_pipeline_error(exc, "hybrid_rag")

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
        return await _build_hybrid_response(payload, question)

    if payload.pipeline == "crag":
        try:
            result = await asyncio.to_thread(run_crag, question, payload.chat_history)
        except Exception as exc:
            _handle_pipeline_error(exc, "crag")

        response_payload = crag_result_to_payload(result)
        if not payload.include_trace:
            response_payload["trace"] = []
        response_payload["meta"] = {
            "history_turns": len(payload.chat_history),
            "trace_included": payload.include_trace,
            "pipeline": payload.pipeline,
        }
        return CRAGResponse.model_validate(response_payload)

    try:
        result = await asyncio.to_thread(run_self_rag, question, payload.chat_history)
    except Exception as exc:
        _handle_pipeline_error(exc, "self_rag")

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

    return await _build_hybrid_response(payload, question)
