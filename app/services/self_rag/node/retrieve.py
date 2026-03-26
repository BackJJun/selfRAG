from langchain_core.prompts import ChatPromptTemplate

from app.prompt.self_rag.retrieval import EVALUATE_RETRIEVED_DOCUMENTS_PROMPT
from app.schemas.rag import GraphState, RetrievalAssessmentResult
from app.services.self_rag.dependencies import get_llm, get_retriever
from app.services.tracing import add_trace
from app.utils.self_rag import format_chat_history, format_documents
from app.core.config import logger


def retrieve(state: GraphState):
    logger.info("Node start | retrieve | current_query=%r", state["current_query"])
    add_trace("retrieve", "Start retrieval", query=state["current_query"])
    documents = get_retriever().invoke(state["current_query"])
    logger.info("Node end   | retrieve | documents=%d", len(documents))
    add_trace(
        "retrieve",
        "Retrieved documents",
        document_count=len(documents),
        sources=[doc.metadata.get("source", "unknown") for doc in documents],
    )
    return {"documents": documents}


def evaluate_retrieved_documents(state: GraphState):
    logger.info(
        "Node start | evaluate_retrieved_documents | question=%r | documents=%d",
        state["question"],
        len(state["documents"]),
    )
    add_trace(
        "retrieval_eval",
        "Evaluate retrieved documents",
        question=state["question"],
        document_count=len(state["documents"]),
    )
    prompt = ChatPromptTemplate.from_template(EVALUATE_RETRIEVED_DOCUMENTS_PROMPT)
    structured_llm = get_llm().with_structured_output(RetrievalAssessmentResult)
    chain = prompt | structured_llm
    result = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "current_query": state["current_query"],
            "context": format_documents(state["documents"]),
        }
    )

    assessments = sorted(result.documents, key=lambda item: item.doc_index)
    selected_indexes = [item.doc_index for item in assessments if item.use]
    if not selected_indexes and state["documents"]:
        selected_indexes = [1]

    filtered_documents = [
        state["documents"][index - 1]
        for index in selected_indexes
        if 1 <= index <= len(state["documents"])
    ]
    discarded_indexes = [
        index
        for index in range(1, len(state["documents"]) + 1)
        if index not in selected_indexes
    ]

    logger.info(
        "Node end   | evaluate_retrieved_documents | kept=%s | discarded=%s",
        selected_indexes,
        discarded_indexes,
    )
    add_trace(
        "retrieval_eval",
        "Retrieved document evaluation complete",
        summary=result.summary,
        used_doc_indexes=selected_indexes,
        discarded_doc_indexes=discarded_indexes,
    )
    return {
        "documents": filtered_documents,
        "retrieval_assessment_summary": result.summary,
        "retrieval_used_doc_indexes": selected_indexes,
        "retrieval_discarded_doc_indexes": discarded_indexes,
        "retrieval_document_assessments": [item.model_dump() for item in assessments],
    }
