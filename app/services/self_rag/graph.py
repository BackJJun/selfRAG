from typing import Any

from langgraph.graph import END, StateGraph

from app.core.config import logger
from app.schemas.rag import GraphState
from app.services.self_rag.dependencies import get_graph_lock, get_llm, get_retriever
from app.services.self_rag.node import (
    evaluate_retrieved_documents,
    generate_answer,
    reflect_on_answer,
    retrieve,
    revise_answer,
    rewrite_query,
    route_after_reflection,
    web_search_node,
)
from app.services.tracing import add_trace

_graph_app: Any | None = None


def build_graph_app():
    """Build and compile the Self-RAG workflow graph."""
    logger.info("[Init] Compile Self-RAG graph")
    add_trace("init", "Compile Self-RAG graph")
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("evaluate_retrieved_documents", evaluate_retrieved_documents)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("reflect_on_answer", reflect_on_answer)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("revise_answer", revise_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "evaluate_retrieved_documents")
    workflow.add_edge("evaluate_retrieved_documents", "generate_answer")
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
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_search_node", "generate_answer")
    workflow.add_edge("revise_answer", END)

    compiled = workflow.compile()
    logger.info("[Init] Self-RAG graph compiled")
    add_trace("init", "Graph ready")
    return compiled


def get_graph_app():
    """Return the compiled graph singleton, building it if needed."""
    global _graph_app
    if _graph_app is None:
        with get_graph_lock():
            if _graph_app is None:
                _graph_app = build_graph_app()
    return _graph_app


def warmup_graph_dependencies() -> None:
    """Warm up shared graph dependencies ahead of the first request."""
    with get_graph_lock():
        get_llm()
        get_retriever()
        global _graph_app
        if _graph_app is None:
            _graph_app = build_graph_app()
