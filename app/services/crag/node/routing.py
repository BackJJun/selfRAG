from app.core.config import MAX_CORRECTION_RETRIES, MAX_RETRIES, logger
from app.schemas.rag import CRAGGraphState
from app.services.tracing import add_trace


# 검색 결과 평가를 바탕으로 refine, 재검색, 웹검색 중 다음 단계를 선택한다.
def route_after_retrieval_assessment(state: CRAGGraphState) -> str:
    logger.info(
        "CRAG route | after_retrieval | quality=%s | score=%d | issue=%s | retry_count=%d | web_search_used=%s",
        state["retrieval_quality"],
        state["retrieval_score"],
        state["retrieval_issue_type"],
        state["retry_count"],
        state["web_search_used"],
    )
    if state["retrieval_quality"] in {"high", "medium"} and state["documents"]:
        add_trace("crag_route", "Proceed to evidence refinement", next_step="refine_evidence")
        return "refine_evidence"

    if state["retrieval_should_use_web"] and not state["web_search_used"]:
        add_trace("crag_route", "Escalate to web search", next_step="web_search_node")
        return "web_search_node"

    if state["retrieval_should_retry"] and state["retry_count"] < MAX_RETRIES:
        add_trace("crag_route", "Retry local retrieval", next_step="rewrite_query")
        return "rewrite_query"

    if not state["web_search_used"]:
        add_trace("crag_route", "Fallback to web search after weak retrieval", next_step="web_search_node")
        return "web_search_node"

    add_trace("crag_route", "Use best available documents and refine evidence", next_step="refine_evidence")
    return "refine_evidence"


# 답변 평가 결과를 corrective action으로 변환해 CRAG 루프를 진행한다.
def route_after_answer_assessment(state: CRAGGraphState) -> str:
    logger.info(
        "CRAG route | after_answer | next_action=%s | retry_count=%d | correction_retry_count=%d | web_search_used=%s",
        state["answer_next_action"],
        state["retry_count"],
        state["correction_retry_count"],
        state["web_search_used"],
    )
    action = state["answer_next_action"]

    if action == "finalize":
        add_trace("crag_route", "Move to final approval", next_step="assess_final_answer")
        return "assess_final_answer"

    if action == "regenerate":
        if state["correction_retry_count"] < MAX_CORRECTION_RETRIES:
            add_trace("crag_route", "Regenerate answer from refined evidence", next_step="regenerate_answer")
            return "regenerate_answer"
        add_trace("crag_route", "Too many regenerate attempts, revise instead", next_step="revise_answer")
        return "revise_answer"

    if action == "rewrite_query":
        if state["retry_count"] < MAX_RETRIES:
            add_trace("crag_route", "Rewrite query from answer assessment", next_step="rewrite_query")
            return "rewrite_query"
        if not state["web_search_used"]:
            add_trace("crag_route", "Retries exhausted, use web search", next_step="web_search_node")
            return "web_search_node"
        add_trace("crag_route", "Retries exhausted after web search, revise answer", next_step="revise_answer")
        return "revise_answer"

    if action == "web_search":
        if not state["web_search_used"]:
            add_trace("crag_route", "Use web search from answer assessment", next_step="web_search_node")
            return "web_search_node"
        add_trace("crag_route", "Web search already used, revise answer", next_step="revise_answer")
        return "revise_answer"

    add_trace("crag_route", "Conservative revision selected", next_step="revise_answer")
    return "revise_answer"


# 최종 확인 결과에 따라 종료하거나 마지막 보수화 단계를 한 번 더 수행한다.
def route_after_final_assessment(state: CRAGGraphState) -> str:
    logger.info(
        "CRAG route | after_final | approved=%s | action=%s | final_revision_count=%d",
        state["final_answer_approved"],
        state["final_answer_action"],
        state["final_revision_count"],
    )
    if state["final_answer_approved"] or state["final_answer_action"] == "end":
        add_trace("crag_route", "Finish CRAG answer", next_step="end")
        return "end"

    if state["final_revision_count"] < 1:
        add_trace("crag_route", "One more conservative revision", next_step="revise_answer")
        return "revise_answer"

    add_trace("crag_route", "Final revision limit reached, finish answer", next_step="end")
    return "end"
