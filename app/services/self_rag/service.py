from typing import cast

from app.core.config import logger
from app.schemas.rag import ChatTurn, GraphState
from app.services.self_rag.graph import get_graph_app
from app.services.shared.runner import run_pipeline
from app.utils.self_rag import make_inputs, result_to_payload


# 사용자 질문과 대화 이력을 받아 Self-RAG 파이프라인 전체를 실행한다.
def run_self_rag(question: str, chat_history: list[ChatTurn]) -> GraphState:
    result = run_pipeline(
        pipeline_name="self-rag",
        run_name="self_rag_graph",
        trace_stage="run",
        start_message="Start Self-RAG run",
        end_message="Self-RAG run completed",
        get_graph_app=get_graph_app,
        make_inputs=make_inputs,
        question=question,
        chat_history=chat_history,
        end_details=lambda result: {
            "decision": result.get("reflection_decision", ""),
            "retry_count": result.get("retry_count", 0),
        },
        log_summary=lambda result, elapsed_ms: logger.info(
            "Run end    | question=%r | elapsed=%.2fs | decision=%s | retry_count=%d",
            question,
            elapsed_ms / 1000,
            result.get("reflection_decision", ""),
            result.get("retry_count", 0),
        ),
    )
    return cast(GraphState, result)


__all__ = ["result_to_payload", "run_self_rag"]
