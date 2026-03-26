from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import logger
from app.prompt.self_rag.revision import REVISE_ANSWER_PROMPT
from app.schemas.rag import GraphState, HybridGraphState
from app.services.shared.dependencies import get_llm
from app.services.tracing import add_trace
from app.utils.self_rag import format_chat_history, format_generation_context

_REVISE_TEMPLATE = ChatPromptTemplate.from_template(REVISE_ANSWER_PROMPT)


# 재검색이 끝난 뒤에도 부족한 답변을 현재 컨텍스트 기준으로 보수적으로 다시 쓴다.
def revise_answer(state: GraphState | HybridGraphState) -> dict[str, Any]:
    logger.info(
        "Node start | revise_answer | question=%r | documents=%d | evidence_count=%d",
        state["question"],
        len(state["documents"]),
        state.get("evidence_count", 0),
    )
    add_trace(
        "revise",
        "Revise final answer after exhausted retrieval",
        question=state["question"],
        document_count=len(state["documents"]),
        evidence_count=state.get("evidence_count", 0),
        issue_source=state["reflection_issue_source"],
    )
    chain = _REVISE_TEMPLATE | get_llm() | StrOutputParser()
    revised_answer = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "context": format_generation_context(state),
            "generation": state["generation"],
        }
    )
    logger.info("Node end   | revise_answer | answer_chars=%d", len(revised_answer))
    add_trace("revise", "Revised final answer ready", answer_chars=len(revised_answer))
    return {
        "generation": revised_answer,
        "reflection_decision": "answer",
    }
