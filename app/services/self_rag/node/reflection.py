from langchain_core.prompts import ChatPromptTemplate

from app.core.config import logger
from app.prompt.self_rag.reflection import REFLECT_ON_ANSWER_PROMPT
from app.schemas.rag import GraphState, HybridGraphState, ReflectionResult
from app.services.shared.dependencies import get_llm
from app.services.tracing import add_trace
from app.utils.self_rag import format_chat_history, format_generation_context


# 생성된 답변이 충분히 근거 기반인지 평가하고 다음 액션에 필요한 신호를 만든다.
def reflect_on_answer(state: GraphState | HybridGraphState):
    logger.info(
        "Node start | reflect_on_answer | current_query=%r | retry_count=%d",
        state["current_query"],
        state["retry_count"],
    )
    add_trace(
        "reflect",
        "Evaluate draft answer",
        current_query=state["current_query"],
        retry_count=state["retry_count"],
        retrieval_summary=state.get("retrieval_assessment_summary", state.get("retrieval_reason", "")),
        evidence_count=state.get("evidence_count", 0),
    )
    prompt = ChatPromptTemplate.from_template(REFLECT_ON_ANSWER_PROMPT)
    structured_llm = get_llm().with_structured_output(ReflectionResult)
    chain = prompt | structured_llm
    result = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "current_query": state["current_query"],
            "context": format_generation_context(state),
            "generation": state["generation"],
        }
    )

    logger.info(
        "Node end   | reflect_on_answer | decision=%s | grounded=%s | complete=%s | relevant=%s | fresh=%s | issue_source=%s | rewritten_query=%r",
        result.decision,
        result.grounded,
        result.complete,
        result.relevant,
        result.fresh,
        result.issue_source,
        (result.rewritten_query or "").strip(),
    )
    add_trace(
        "reflect",
        "Reflection complete",
        decision=result.decision,
        grounded=result.grounded,
        complete=result.complete,
        relevant=result.relevant,
        fresh=result.fresh,
        issue_source=result.issue_source,
        rationale=result.rationale,
        rewritten_query=(result.rewritten_query or "").strip(),
    )
    return {
        "reflection_decision": result.decision,
        "rewritten_query": (result.rewritten_query or "").strip(),
        "reflection_grounded": result.grounded,
        "reflection_complete": result.complete,
        "reflection_relevant": result.relevant,
        "reflection_fresh": result.fresh,
        "reflection_issue_source": result.issue_source,
        "reflection_rationale": result.rationale,
    }
