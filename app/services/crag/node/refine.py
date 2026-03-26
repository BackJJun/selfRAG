from langchain_core.prompts import ChatPromptTemplate

from app.core.config import logger
from app.prompt.crag.refine import CRAG_REFINE_EVIDENCE_PROMPT
from app.schemas.rag import CRAGGraphState, CRAGRefineEvidenceResult
from app.services.shared.dependencies import get_llm
from app.services.tracing import add_trace
from app.utils.crag import format_chat_history, format_documents


# 평가를 통과한 문서에서 질문과 직접 관련된 evidence set만 구조화해 추출한다.
def refine_evidence(state: CRAGGraphState):
    logger.info(
        "CRAG node  | refine_evidence | question=%r | documents=%d",
        state["question"],
        len(state["documents"]),
    )
    add_trace(
        "crag_refine",
        "Refine evidence from filtered documents",
        question=state["question"],
        document_count=len(state["documents"]),
        retrieval_quality=state["retrieval_quality"],
    )
    prompt = ChatPromptTemplate.from_template(CRAG_REFINE_EVIDENCE_PROMPT)
    structured_llm = get_llm().with_structured_output(CRAGRefineEvidenceResult)
    chain = prompt | structured_llm
    result = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "current_query": state["current_query"],
            "retrieval_summary": state["retrieval_reason"],
            "context": format_documents(state["documents"]),
        }
    )
    items = [item.model_dump() for item in result.items]
    add_trace(
        "crag_refine",
        "Evidence refinement complete",
        quality=result.quality,
        evidence_count=len(items),
        summary=result.summary,
    )
    return {
        "refined_evidence": items,
        "refine_summary": result.summary,
        "refine_quality": result.quality,
        "evidence_count": len(items),
    }
