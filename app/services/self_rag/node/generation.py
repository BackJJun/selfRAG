from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import logger
from app.prompt.self_rag.generation import GENERATE_ANSWER_PROMPT
from app.schemas.chat import GraphState
from app.services.self_rag.dependencies import get_llm
from app.services.tracing import add_trace
from app.utils.self_rag import format_chat_history, format_documents


def generate_answer(state: GraphState):
    logger.info(
        "Node start | generate_answer | question=%r | history_turns=%d | documents=%d",
        state["question"],
        len(state["chat_history"]),
        len(state["documents"]),
    )
    add_trace(
        "generate",
        "Generate answer draft",
        question=state["question"],
        history_turns=len(state["chat_history"]),
        document_count=len(state["documents"]),
        used_doc_indexes=state["retrieval_used_doc_indexes"],
    )
    prompt = ChatPromptTemplate.from_template(GENERATE_ANSWER_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
    generation = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "context": format_documents(state["documents"]),
        }
    )
    logger.info("Node end   | generate_answer | answer_chars=%d", len(generation))
    add_trace("generate", "Draft answer ready", answer_chars=len(generation))
    return {"generation": generation}
