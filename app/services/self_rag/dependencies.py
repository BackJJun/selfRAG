import threading

from langchain_openai import ChatOpenAI

from app.core.config import OPENAI_LLM_MODEL, logger
from app.services.retriever import LocalVectorRetriever, build_retriever
from app.services.tracing import add_trace

_graph_lock = threading.Lock()
_retriever: LocalVectorRetriever | None = None
_llm: ChatOpenAI | None = None


# 그래프(LangGraph) 초기화 및 리소스 생성 시, 레이스 컨디션을 방지하기 위한 전역 스레드 락을 반환한다.
def get_graph_lock() -> threading.Lock:
    """그래프 초기화 동기화를 위한 락 객체를 반환합니다."""
    return _graph_lock


# LocalVectorRetriever 인스턴스를 싱글톤 패턴으로 관리한다. 
# 최초 호출 시에만 벡터DB로부터 리트리버를 빌드하며, 이후 호출 시에는 캐시된 인스턴스를 반환한다.
def get_retriever() -> LocalVectorRetriever:
    """전역 리트리버 인스턴스를 반환하며, 없을 경우 초기화하여 생성합니다."""
    global _retriever
    if _retriever is None:
        logger.info("[Init] Build retriever")
        add_trace("init", "Build retriever")
        _retriever = build_retriever()
        add_trace("init", "Retriever ready", chunk_count=len(_retriever.chunks))
    return _retriever


# ChatOpenAI 클라이언트를 싱글톤 패턴으로 관리한다.
# 설정된 모델명(OPENAI_LLM_MODEL)과 기본 temperature(0) 값을 사용하여 LLM 인스턴스를 생성하고 재사용한다.
def get_llm() -> ChatOpenAI:
    """전역 LLM 클라이언트를 반환하며, 없을 경우 초기화하여 생성합니다."""
    global _llm
    if _llm is None:
        logger.info("[Init] Create chat model | model=%s", OPENAI_LLM_MODEL)
        add_trace("init", "Create chat model", model=OPENAI_LLM_MODEL)
        _llm = ChatOpenAI(model=OPENAI_LLM_MODEL, temperature=0)
    return _llm
