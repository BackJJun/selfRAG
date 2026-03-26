from typing import Any

from langgraph.graph import END, StateGraph

from app.core.config import MAX_RETRIES, logger
from app.schemas.rag import HybridGraphState
from app.services.shared.dependencies import get_graph_lock, get_llm, get_retriever
from app.services.crag.node.correction import rewrite_query, web_search_node
from app.services.crag.node.refine import refine_evidence
from app.services.crag.node.retrieve import assess_retrieval_quality, retrieve
from app.services.self_rag.node.generation import generate_answer
from app.services.self_rag.node.reflection import reflect_on_answer
from app.services.self_rag.node.revision import revise_answer
from app.services.tracing import add_trace

_graph_app: Any | None = None


# 검색 품질 평가 결과를 바탕으로 다음 retrieval 단계를 결정한다.
def route_after_retrieval_assessment(state: HybridGraphState):
    if state["retrieval_quality"] in {"high", "medium"} and state["documents"]:
        add_trace("hybrid_route", "Proceed to evidence refinement", next_step="refine_evidence")
        return "refine_evidence"

    if state["retrieval_should_use_web"] and not state["web_search_used"]:
        add_trace("hybrid_route", "Escalate to web search", next_step="web_search_node")
        return "web_search_node"

    if state["retrieval_should_retry"] and state["retry_count"] < MAX_RETRIES:
        add_trace("hybrid_route", "Retry local retrieval", next_step="rewrite_query")
        return "rewrite_query"

    if not state["web_search_used"]:
        add_trace("hybrid_route", "Fallback to web search", next_step="web_search_node")
        return "web_search_node"

    add_trace("hybrid_route", "Use best available evidence", next_step="refine_evidence")
    return "refine_evidence"


# Self-RAG reflection 결과를 바탕으로 retrieval corrective action 또는 답변 보수화를 선택한다.
def route_after_reflection(state: HybridGraphState):
    if state["reflection_decision"] == "answer":
        add_trace("hybrid_route", "Finish with current answer", next_step="end")
        return "end"

    if state["reflection_issue_source"] == "answer_problem":
        add_trace("hybrid_route", "Revise answer after reflection", next_step="revise_answer")
        return "revise_answer"

    if (not state["reflection_fresh"]) or state["reflection_issue_source"] == "freshness_problem":
        if not state["web_search_used"]:
            add_trace("hybrid_route", "Use web search for freshness", next_step="web_search_node")
            return "web_search_node"
        add_trace("hybrid_route", "Web search already used, revise answer", next_step="revise_answer")
        return "revise_answer"

    if state["retry_count"] < MAX_RETRIES:
        add_trace("hybrid_route", "Retry retrieval from reflection", next_step="rewrite_query")
        return "rewrite_query"

    if not state["web_search_used"]:
        add_trace("hybrid_route", "Retries exhausted, use web search", next_step="web_search_node")
        return "web_search_node"

    add_trace("hybrid_route", "Retries exhausted after web search, revise answer", next_step="revise_answer")
    return "revise_answer"


# 하이브리드 RAG 그래프를 조립해 CRAG 검색 단계와 Self-RAG 답변 단계를 연결한다.
def build_graph_app():
    logger.info("[Init] Compile Hybrid-RAG graph")
    add_trace("hybrid_init", "Compile Hybrid-RAG graph")
    workflow = StateGraph(HybridGraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("assess_retrieval_quality", assess_retrieval_quality)
    workflow.add_node("refine_evidence", refine_evidence)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("reflect_on_answer", reflect_on_answer)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("revise_answer", revise_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "assess_retrieval_quality")
    workflow.add_conditional_edges(
        "assess_retrieval_quality",
        route_after_retrieval_assessment,
        {
            "refine_evidence": "refine_evidence",
            "rewrite_query": "rewrite_query",
            "web_search_node": "web_search_node",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_search_node", "assess_retrieval_quality")
    workflow.add_edge("refine_evidence", "generate_answer")
    workflow.add_edge("generate_answer", "reflect_on_answer")
    workflow.add_conditional_edges(
        "reflect_on_answer",
        route_after_reflection,
        {
            "end": END,
            "rewrite_query": "rewrite_query",
            "web_search_node": "web_search_node",
            "revise_answer": "revise_answer",
        },
    )
    workflow.add_edge("revise_answer", END)

    compiled = workflow.compile()
    logger.info("[Init] Hybrid-RAG graph compiled")
    add_trace("hybrid_init", "Hybrid-RAG graph ready")
    return compiled


# 하이브리드 그래프 싱글톤을 반환하고 없으면 한 번만 생성한다.
def get_graph_app():
    global _graph_app
    if _graph_app is None:
        with get_graph_lock():
            if _graph_app is None:
                _graph_app = build_graph_app()
    return _graph_app


# 첫 요청 전에 하이브리드 RAG가 공용 의존성과 그래프를 미리 준비하도록 워밍업한다.
def warmup_graph_dependencies() -> None:
    with get_graph_lock():
        get_llm()
        get_retriever()
        global _graph_app
        if _graph_app is None:
            _graph_app = build_graph_app()
