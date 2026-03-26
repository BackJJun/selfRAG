from typing import cast

from app.core.config import logger
from app.schemas.rag import ChatTurn, HybridGraphState
from app.services.hybrid_rag.graph import get_graph_app
from app.services.shared.runner import run_pipeline
from app.utils.hybrid_rag import make_inputs, result_to_payload as build_payload


# 사용자 질문과 대화 이력을 받아 Hybrid-RAG 파이프라인 전체를 실행한다.
def run_hybrid_rag(question: str, chat_history: list[ChatTurn]) -> HybridGraphState:
    result = run_pipeline(
        pipeline_name="hybrid-rag",
        run_name="hybrid_rag_graph",
        trace_stage="hybrid_run",
        start_message="Start Hybrid-RAG run",
        end_message="Hybrid-RAG run completed",
        get_graph_app=get_graph_app,
        make_inputs=make_inputs,
        question=question,
        chat_history=chat_history,
        end_details=lambda result: {
            "decision": result.get("reflection_decision", ""),
            "retry_count": result.get("retry_count", 0),
            "evidence_count": result.get("evidence_count", 0),
        },
        log_summary=lambda result, elapsed_ms: logger.info(
            "Hybrid run end | question=%r | elapsed=%.2fs | decision=%s | retry_count=%d",
            question,
            elapsed_ms / 1000,
            result.get("reflection_decision", ""),
            result.get("retry_count", 0),
        ),
    )
    return cast(HybridGraphState, result)


# 하이브리드 RAG 결과를 외부 응답용 payload로 변환한다.
def result_to_payload(result: HybridGraphState) -> dict:
    return build_payload(result)
