from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import logger
from app.prompt.common import REWRITE_QUERY_PROMPT
from app.schemas.rag import GraphState
from app.services.shared.dependencies import get_llm
from app.services.tracing import add_trace
from app.utils.self_rag import format_chat_history


def rewrite_query(state: GraphState):
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
    suggested_query = state["rewritten_query"].strip()
    if suggested_query:
        logger.info("Node end   | rewrite_query | using_reflection_query=%r", suggested_query)
        add_trace("rewrite", "Use reflection query", rewritten_query=suggested_query)
        return {
            "current_query": suggested_query,
            "retry_count": state["retry_count"] + 1,
        }

    prompt = ChatPromptTemplate.from_template(REWRITE_QUERY_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
    better_query = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "current_query": state["current_query"],
        }
    ).strip()

    logger.info("Node end   | rewrite_query | generated_query=%r", better_query)
    add_trace("rewrite", "Generated new query", rewritten_query=better_query)
    return {
        "current_query": better_query,
        "retry_count": state["retry_count"] + 1,
    }
