from typing import Any

from langgraph.graph import END, StateGraph

from app.core.config import logger
from app.schemas.rag import CRAGGraphState
from app.services.crag.dependencies import get_graph_lock, get_llm, get_retriever
from app.services.crag.node import (
    assess_answer_quality,
    assess_final_answer,
    assess_retrieval_quality,
    generate_answer,
    refine_evidence,
    regenerate_answer,
    retrieve,
    revise_answer,
    rewrite_query,
    route_after_answer_assessment,
    route_after_final_assessment,
    route_after_retrieval_assessment,
    web_search_node,
)
from app.services.tracing import add_trace

_graph_app: Any | None = None


# CRAG corrective loop 그래프를 조립하고 재사용 가능한 실행 객체로 컴파일한다.
def build_graph_app():
    logger.info("[Init] Compile CRAG graph")
    add_trace("crag_init", "Compile CRAG graph")
    workflow = StateGraph(CRAGGraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("assess_retrieval_quality", assess_retrieval_quality)
    workflow.add_node("refine_evidence", refine_evidence)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("regenerate_answer", regenerate_answer)
    workflow.add_node("assess_answer_quality", assess_answer_quality)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("revise_answer", revise_answer)
    workflow.add_node("assess_final_answer", assess_final_answer)

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
    workflow.add_edge("generate_answer", "assess_answer_quality")
    workflow.add_edge("regenerate_answer", "assess_answer_quality")
    workflow.add_conditional_edges(
        "assess_answer_quality",
        route_after_answer_assessment,
        {
            "assess_final_answer": "assess_final_answer",
            "regenerate_answer": "regenerate_answer",
            "rewrite_query": "rewrite_query",
            "web_search_node": "web_search_node",
            "revise_answer": "revise_answer",
        },
    )
    workflow.add_edge("revise_answer", "assess_final_answer")
    workflow.add_conditional_edges(
        "assess_final_answer",
        route_after_final_assessment,
        {
            "revise_answer": "revise_answer",
            "end": END,
        },
    )

    compiled = workflow.compile()
    logger.info("[Init] CRAG graph compiled")
    add_trace("crag_init", "CRAG graph ready")
    return compiled


# CRAG 그래프 싱글톤을 반환하고 아직 없으면 한 번만 안전하게 생성한다.
def get_graph_app():
    global _graph_app
    if _graph_app is None:
        with get_graph_lock():
            if _graph_app is None:
                _graph_app = build_graph_app()
    return _graph_app


# 첫 요청 전에 CRAG가 공용 의존성과 그래프를 미리 준비하도록 워밍업한다.
def warmup_graph_dependencies() -> None:
    with get_graph_lock():
        get_llm()
        get_retriever()
        global _graph_app
        if _graph_app is None:
            _graph_app = build_graph_app()
