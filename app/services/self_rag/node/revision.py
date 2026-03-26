from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import logger
from app.prompt.self_rag.revision import REVISE_ANSWER_PROMPT
from app.schemas.chat import GraphState
from app.services.self_rag.dependencies import get_llm
from app.services.tracing import add_trace
from app.utils.self_rag import format_chat_history, format_documents


def revise_answer(state: GraphState):
    logger.info(
        "Node start | revise_answer | question=%r | documents=%d",
        state["question"],
        len(state["documents"]),
    )
    add_trace(
        "revise",
        "Revise final answer after exhausted retrieval",
        question=state["question"],
        document_count=len(state["documents"]),
        issue_source=state["reflection_issue_source"],
    )
    prompt = ChatPromptTemplate.from_template(REVISE_ANSWER_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
    revised_answer = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "context": format_documents(state["documents"]),
            "generation": state["generation"],
        }
    )
    logger.info("Node end   | revise_answer | answer_chars=%d", len(revised_answer))
    add_trace("revise", "Revised final answer ready", answer_chars=len(revised_answer))
    return {
        "generation": revised_answer,
        "reflection_decision": "answer",
    }
