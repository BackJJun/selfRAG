from app.core.config import MAX_RETRIES, logger
from app.schemas.chat import GraphState
from app.services.tracing import add_trace


def route_after_reflection(state: GraphState):
    logger.info(
        "Router     | route_after_reflection | decision=%s | retry_count=%d | web_search_used=%s | issue_source=%s | fresh=%s",
        state["reflection_decision"],
        state["retry_count"],
        state["web_search_used"],
        state["reflection_issue_source"],
        state["reflection_fresh"],
    )

    if state["reflection_decision"] == "answer":
        logger.info("Router     | route_after_reflection | next=end")
        add_trace("route", "Finish with current answer", next_step="end")
        return "end"

    if state["web_search_used"]:
        logger.info("Router     | route_after_reflection | next=revise_answer")
        add_trace(
            "route",
            "Revise answer after web-search fallback instead of looping again",
            next_step="revise_answer",
            reason="web_search_already_used",
        )
        return "revise_answer"

    if (not state["reflection_fresh"]) or state["reflection_issue_source"] == "freshness_problem":
        logger.info("Router     | route_after_reflection | next=web_search_node")
        add_trace(
            "route",
            "Go to web search because freshness is insufficient",
            next_step="web_search_node",
        )
        return "web_search_node"

    if state["reflection_issue_source"] == "answer_problem":
        logger.info("Router     | route_after_reflection | next=revise_answer")
        add_trace(
            "route",
            "Revise answer because retrieval looks usable but synthesis is weak",
            next_step="revise_answer",
        )
        return "revise_answer"

    if state["retry_count"] < MAX_RETRIES and state["reflection_issue_source"] in {
        "query_problem",
        "retrieval_problem",
        "mixed",
        "none",
    }:
        logger.info("Router     | route_after_reflection | next=rewrite_query")
        add_trace(
            "route",
            "Retry retrieval with rewritten query",
            next_step="rewrite_query",
            issue_source=state["reflection_issue_source"],
        )
        return "rewrite_query"

    logger.info("Router     | route_after_reflection | next=web_search_node")
    add_trace("route", "Fallback to web search", next_step="web_search_node")
    return "web_search_node"
