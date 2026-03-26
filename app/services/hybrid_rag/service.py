import time

from langchain_core.runnables import RunnableConfig

from app.core.config import LANGSMITH_PROJECT, MAX_RETRIES, logger
from app.schemas.rag import ChatTurn, HybridGraphState
from app.services.hybrid_rag.graph import get_graph_app
from app.services.tracing import add_trace, begin_trace, end_trace
from app.utils.hybrid_rag import make_inputs, result_to_payload as build_payload


# 사용자 질문과 대화 이력을 받아 Hybrid-RAG 파이프라인 전체를 실행한다.
def run_hybrid_rag(question: str, chat_history: list[ChatTurn]) -> HybridGraphState:
    trace_events, trace_token, started_token, started_at = begin_trace()

    try:
        logger.info(
            "Hybrid run start | question=%r | history_turns=%d",
            question,
            len(chat_history),
        )
        add_trace("hybrid_run", "Start Hybrid-RAG run", question=question, history_turns=len(chat_history))
        invoke_config: RunnableConfig = {
            "run_name": "hybrid_rag_graph",
            "tags": ["hybrid-rag", LANGSMITH_PROJECT],
            "metadata": {
                "question": question,
                "history_turns": len(chat_history),
                "max_retries": MAX_RETRIES,
            },
        }
        result = get_graph_app().invoke(make_inputs(question, chat_history), config=invoke_config)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "Hybrid run end | question=%r | elapsed=%.2fs | decision=%s | retry_count=%d",
            question,
            elapsed_ms / 1000,
            result.get("reflection_decision", ""),
            result.get("retry_count", 0),
        )
        add_trace(
            "hybrid_run",
            "Hybrid-RAG run completed",
            elapsed_ms=elapsed_ms,
            decision=result.get("reflection_decision", ""),
            retry_count=result.get("retry_count", 0),
            evidence_count=result.get("evidence_count", 0),
        )
        result["trace"] = list(trace_events)
        return result
    finally:
        end_trace(trace_token, started_token)


# 하이브리드 RAG 결과를 외부 응답용 payload로 변환한다.
def result_to_payload(result: HybridGraphState) -> dict:
    return build_payload(result)
