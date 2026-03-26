from langchain_core.prompts import ChatPromptTemplate

from app.core.config import logger
from app.prompt.crag.retrieval import CRAG_RETRIEVAL_QUALITY_PROMPT
from app.schemas.rag import CRAGGraphState, CRAGRetrievalAssessmentResult
from app.services.shared.dependencies import get_llm, get_retriever
from app.services.tracing import add_trace
from app.utils.crag import format_chat_history, format_documents


# 현재 질의로 로컬 리트리버를 호출해 CRAG의 1차 검색 결과를 수집한다.
def retrieve(state: CRAGGraphState):
    logger.info("CRAG node  | retrieve | current_query=%r", state["current_query"])
    add_trace("crag_retrieve", "CRAG retrieval started", query=state["current_query"])
    documents = get_retriever().invoke(state["current_query"])
    add_trace(
        "crag_retrieve",
        "CRAG retrieval finished",
        document_count=len(documents),
        sources=[doc.metadata.get("source", "unknown") for doc in documents],
    )
    return {"documents": documents}


# 검색 품질과 문서별 관련도를 함께 평가해 이후 corrective action의 기준 신호를 만든다.
def assess_retrieval_quality(state: CRAGGraphState):
    logger.info(
        "CRAG node  | assess_retrieval_quality | question=%r | documents=%d",
        state["question"],
        len(state["documents"]),
    )
    add_trace(
        "crag_retrieval_reflect",
        "Assess retrieval quality",
        question=state["question"],
        document_count=len(state["documents"]),
        retry_count=state["retry_count"],
    )
    prompt = ChatPromptTemplate.from_template(CRAG_RETRIEVAL_QUALITY_PROMPT)
    structured_llm = get_llm().with_structured_output(CRAGRetrievalAssessmentResult)
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

    add_trace(
        "crag_retrieval_reflect",
        "Retrieval quality assessed",
        quality=result.quality,
        score=result.score,
        issue_type=result.issue_type,
        should_retry=result.should_retry_retrieval,
        should_use_web=result.should_use_web,
        used_doc_indexes=selected_indexes,
        discarded_doc_indexes=discarded_indexes,
    )
    return {
        "documents": filtered_documents,
        "rewritten_query": (result.rewritten_query or "").strip(),
        "retrieval_quality": result.quality,
        "retrieval_score": result.score,
        "retrieval_issue_type": result.issue_type,
        "retrieval_reason": result.summary,
        "retrieval_should_retry": result.should_retry_retrieval,
        "retrieval_should_use_web": result.should_use_web,
        "retrieval_used_doc_indexes": selected_indexes,
        "retrieval_discarded_doc_indexes": discarded_indexes,
        "retrieval_document_assessments": [item.model_dump() for item in assessments],
    }
