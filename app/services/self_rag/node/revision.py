from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import logger
from app.prompt.self_rag.revision import REVISE_ANSWER_PROMPT
from app.schemas.rag import GraphState, HybridGraphState
from app.services.shared.dependencies import get_llm
from app.services.tracing import add_trace
from app.utils.self_rag import format_chat_history, format_generation_context


# 재검색이 끝났거나 보수화가 필요할 때 현재 컨텍스트 기준으로 최종 답변을 다듬는다.
def revise_answer(state: GraphState | HybridGraphState):
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
    prompt = ChatPromptTemplate.from_template(REVISE_ANSWER_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
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
