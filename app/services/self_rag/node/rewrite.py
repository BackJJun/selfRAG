from typing import Any

from app.core.config import logger
from app.schemas.rag import GraphState
from app.services.shared.query_rewrite import resolve_rewritten_query
from app.services.tracing import add_trace


# 현재 질문과 반성 결과를 바탕으로 다음 로컬 검색 질의를 만든다.
def rewrite_query(state: GraphState) -> dict[str, Any]:
    logger.info(
        "Node start | rewrite_query | previous_query=%r | retry_count=%d",
        state["current_query"],
        state["retry_count"],
    )
    add_trace(
        "rewrite",
        "Rewrite retrieval query",
        previous_query=state["current_query"],
        retry_count=state["retry_count"],
    )
    better_query = resolve_rewritten_query(
        question=state["question"],
        chat_history=state["chat_history"],
        current_query=state["current_query"],
        suggested_query=state["rewritten_query"],
    )
    logger.info("Node end   | rewrite_query | generated_query=%r", better_query)
    add_trace("rewrite", "Generated new query", rewritten_query=better_query)
    return {
        "current_query": better_query,
        "retry_count": state["retry_count"] + 1,
    }
