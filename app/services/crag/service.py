from typing import cast

from app.core.config import logger
from app.schemas.rag import CRAGGraphState, ChatTurn
from app.services.crag.graph import get_graph_app
from app.services.shared.runner import run_pipeline
from app.utils.crag import make_inputs, result_to_payload as build_payload


# 사용자 질문과 대화 이력을 받아 CRAG corrective loop 전체를 실행한다.
def run_crag(question: str, chat_history: list[ChatTurn]) -> CRAGGraphState:
    result = run_pipeline(
        pipeline_name="crag",
        run_name="crag_graph",
        trace_stage="crag_run",
        start_message="Start CRAG run",
        end_message="CRAG run completed",
        get_graph_app=get_graph_app,
        make_inputs=make_inputs,
        question=question,
        chat_history=chat_history,
        end_details=lambda result: {
            "retry_count": result.get("retry_count", 0),
            "correction_retry_count": result.get("correction_retry_count", 0),
            "evidence_count": result.get("evidence_count", 0),
        },
        log_summary=lambda result, elapsed_ms: logger.info(
            "CRAG end   | question=%r | elapsed=%.2fs | retry_count=%d | correction_retry_count=%d",
            question,
            elapsed_ms / 1000,
            result.get("retry_count", 0),
            result.get("correction_retry_count", 0),
        ),
    )
    return cast(CRAGGraphState, result)


# CRAG 그래프 결과를 외부 응답용 payload로 변환한다.
def result_to_payload(result: CRAGGraphState) -> dict:
    return build_payload(result)
