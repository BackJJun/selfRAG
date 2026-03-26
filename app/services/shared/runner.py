import time
from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.core.config import LANGSMITH_PROJECT, MAX_RETRIES
from app.schemas.rag import ChatTurn
from app.services.tracing import add_trace, begin_trace, end_trace


# 공통 파이프라인 실행 절차를 재사용해 트레이싱, 로깅, 그래프 invoke를 한 번에 처리한다.
def run_pipeline(
    *,
    pipeline_name: str,
    run_name: str,
    trace_stage: str,
    start_message: str,
    end_message: str,
    get_graph_app: Callable[[], Any],
    make_inputs: Callable[[str, list[ChatTurn]], dict[str, Any]],
    question: str,
    chat_history: list[ChatTurn],
    end_details: Callable[[dict[str, Any]], dict[str, Any]],
    log_summary: Callable[[dict[str, Any], int], None],
) -> dict[str, Any]:
    trace_events, trace_token, started_token, started_at = begin_trace()

    try:
        add_trace(trace_stage, start_message, question=question, history_turns=len(chat_history))
        invoke_config: RunnableConfig = {
            "run_name": run_name,
            "tags": [pipeline_name, LANGSMITH_PROJECT],
            "metadata": {
                "question": question,
                "history_turns": len(chat_history),
                "max_retries": MAX_RETRIES,
            },
        }
        result = get_graph_app().invoke(make_inputs(question, chat_history), config=invoke_config)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        log_summary(result, elapsed_ms)
        add_trace(trace_stage, end_message, elapsed_ms=elapsed_ms, **end_details(result))
        result["trace"] = list(trace_events)
        return result
    finally:
        end_trace(trace_token, started_token)
