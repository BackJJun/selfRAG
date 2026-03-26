import time

from langchain_core.runnables import RunnableConfig

from app.core.config import LANGSMITH_PROJECT, MAX_RETRIES, logger
from app.schemas.chat import ChatTurn, GraphState
from app.services.self_rag.graph import get_graph_app
from app.services.self_rag.tracing import add_trace, begin_trace, end_trace
from app.services.self_rag.utils import make_inputs, result_to_payload


# 사용자 질문과 이전 대화 기록을 바탕으로 Self-RAG 프로세스를 총괄 실행한다.
# 1. 실행 과정을 기록하기 위한 트레이싱(Tracing)을 시작한다.
# 2. 질문과 대화 기록을 그래프 입력 형식으로 변환한다.
# 3. 컴파일된 LangGraph 앱을 호출하여 비즈니스 로직(검색-생성-반성)을 수행한다.
# 4. 종료 후 수행 시간 및 상세 트레이스 정보를 포함한 최종 상태(GraphState)를 반환한다.
def run_self_rag(question: str, chat_history: list[ChatTurn]) -> GraphState:
    """사용자 질문에 대해 Self-RAG 알고리즘을 적용하여 최적의 답변을 생성합니다."""

    trace_events, trace_token, started_token, started_at = begin_trace()

    try:
        logger.info(
            "Run start  | question=%r | history_turns=%d",
            question,
            len(chat_history),
        )
        add_trace("run", "Start Self-RAG run", question=question, history_turns=len(chat_history))
        invoke_config: RunnableConfig = {
            "run_name": "self_rag_graph",
            "tags": ["self-rag", LANGSMITH_PROJECT],
            "metadata": {
                "question": question,
                "history_turns": len(chat_history),
                "max_retries": MAX_RETRIES,
            },
        }
        result = get_graph_app().invoke(make_inputs(question, chat_history), config=invoke_config)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "Run end    | question=%r | elapsed=%.2fs | decision=%s | retry_count=%d",
            question,
            elapsed_ms / 1000,
            result.get("reflection_decision", ""),
            result.get("retry_count", 0),
        )
        add_trace(
            "run",
            "Self-RAG run completed",
            elapsed_ms=elapsed_ms,
            decision=result.get("reflection_decision", ""),
            retry_count=result.get("retry_count", 0),
        )
        result["trace"] = list(trace_events)
        return result
    finally:
        end_trace(trace_token, started_token)


__all__ = ["result_to_payload", "run_self_rag"]
