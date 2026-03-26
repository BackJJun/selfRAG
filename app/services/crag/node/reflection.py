from langchain_core.prompts import ChatPromptTemplate

from app.core.config import logger
from app.prompt.crag.reflection import CRAG_ANSWER_QUALITY_PROMPT, CRAG_FINAL_ANSWER_PROMPT
from app.schemas.rag import CRAGAnswerAssessmentResult, CRAGFinalAssessmentResult, CRAGGraphState
from app.services.crag.dependencies import get_llm
from app.services.tracing import add_trace
from app.utils.crag import format_chat_history, format_refined_evidence


# 생성된 답변을 refined evidence 기준으로 평가해 다음 corrective action을 결정한다.
def assess_answer_quality(state: CRAGGraphState):
    logger.info(
        "CRAG node  | assess_answer_quality | current_query=%r | correction_retry_count=%d",
        state["current_query"],
        state["correction_retry_count"],
    )
    add_trace(
        "crag_answer_reflect",
        "Assess answer quality against refined evidence",
        current_query=state["current_query"],
        correction_retry_count=state["correction_retry_count"],
        evidence_count=state["evidence_count"],
    )
    prompt = ChatPromptTemplate.from_template(CRAG_ANSWER_QUALITY_PROMPT)
    structured_llm = get_llm().with_structured_output(CRAGAnswerAssessmentResult)
    chain = prompt | structured_llm
    result = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "current_query": state["current_query"],
            "context": format_refined_evidence(state["refined_evidence"]),
            "generation": state["generation"],
        }
    )
    add_trace(
        "crag_answer_reflect",
        "Answer quality assessed",
        grounded=result.grounded,
        complete=result.complete,
        relevant=result.relevant,
        next_action=result.next_action,
        rationale=result.rationale,
    )
    return {
        "answer_grounded": result.grounded,
        "answer_complete": result.complete,
        "answer_relevant": result.relevant,
        "answer_rationale": result.rationale,
        "answer_next_action": result.next_action,
        "answer_missing_points": result.missing_points,
        "answer_unsupported_claims": result.unsupported_claims,
        "rewritten_query": (result.rewritten_query or "").strip(),
    }


# 반환 직전 답변을 한 번 더 점검해 종료 또는 마지막 보수화 여부를 결정한다.
def assess_final_answer(state: CRAGGraphState):
    logger.info(
        "CRAG node  | assess_final_answer | final_revision_count=%d",
        state["final_revision_count"],
    )
    add_trace(
        "crag_final_reflect",
        "Assess final answer approval",
        final_revision_count=state["final_revision_count"],
        answer_next_action=state["answer_next_action"],
    )
    prompt = ChatPromptTemplate.from_template(CRAG_FINAL_ANSWER_PROMPT)
    structured_llm = get_llm().with_structured_output(CRAGFinalAssessmentResult)
    chain = prompt | structured_llm
    result = chain.invoke(
        {
            "question": state["question"],
            "context": format_refined_evidence(state["refined_evidence"]),
            "generation": state["generation"],
            "answer_assessment": state["answer_rationale"],
        }
    )
    add_trace(
        "crag_final_reflect",
        "Final approval assessed",
        approved=result.approved,
        action=result.action,
        rationale=result.rationale,
    )
    return {
        "final_answer_approved": result.approved,
        "final_answer_action": result.action,
        "final_answer_rationale": result.rationale,
    }
