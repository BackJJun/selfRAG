from typing import Any

from langgraph.graph import END, StateGraph

from app.core.config import logger
from app.schemas.chat import GraphState
from app.services.self_rag.dependencies import get_graph_lock, get_llm, get_retriever
from app.services.self_rag.nodes import (
    generate_answer,
    reflect_on_answer,
    retrieve,
    revise_answer,
    rewrite_query,
    route_after_reflection,
    web_search_node,
)
from app.services.self_rag.tracing import add_trace

_graph_app: Any | None = None


# Self-RAG(Self-Reflective Retrieval-Augmented Generation) 전체 워크플로우를 정의하는 그래프를 구성한다.
# Retrieve, Generate, Reflect 등의 노드를 정의하고, 반성 결과에 따른 조건부 엣지를 포함한 전체 흐름을 컴파일한다.
def build_graph_app():
    """Self-RAG 워크플로우 그래프를 생성하고 컴파일하여 반환합니다."""
    logger.info("[Init] Compile Self-RAG graph")
    add_trace("init", "Compile Self-RAG graph")
    workflow = StateGraph(GraphState)
    
    # 각 처리 단계(Node) 등록
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("reflect_on_answer", reflect_on_answer)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("revise_answer", revise_answer)
    
    # 그래프 흐름 정의
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", "reflect_on_answer")
    
    # 반성(Reflection) 결과에 따른 조건부 분기 설정
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
    
    # 그래프 컴파일
    compiled = workflow.compile()
    logger.info("[Init] Self-RAG graph compiled")
    add_trace("init", "Graph ready")
    return compiled


# 컴파일된 그래프 앱 인스턴스를 싱글톤 패턴으로 관리한다.
# 스레드 락을 사용하여 멀티스레드 환경에서도 안전하게 단 한 번만 생성되도록 보장한다.
def get_graph_app():
    """전역 그래프 앱 인스턴스를 안전하게 반환하며, 필요 시 지연 초기화합니다."""
    global _graph_app
    if _graph_app is None:
        with get_graph_lock():
            if _graph_app is None:
                _graph_app = build_graph_app()
    return _graph_app


# 애플리케이션 시작 시 호출되어 LLM, 리트리버 및 그래프 앱을 미리 로드한다.
# 첫 번째 요청 시의 지연 시간(Cold Start)을 방지하기 위해 사용된다.
def warmup_graph_dependencies() -> None:
    """모든 의존성 리소스를 미리 초기화하여 준비 상태로 만듭니다."""
    with get_graph_lock():
        get_llm()
        get_retriever()
        global _graph_app
        if _graph_app is None:
            _graph_app = build_graph_app()
