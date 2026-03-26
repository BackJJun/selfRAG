import threading

from langchain_openai import ChatOpenAI

from app.core.config import OPENAI_LLM_MODEL, logger
from app.services.retriever import LocalVectorRetriever, build_retriever
from app.services.tracing import add_trace

_graph_lock = threading.Lock()
_retriever: LocalVectorRetriever | None = None
_llm: ChatOpenAI | None = None


# 그래프 초기화 시 동시 빌드를 막기 위한 공용 락 객체를 반환한다.
def get_graph_lock() -> threading.Lock:
    return _graph_lock


# 공용 리트리버 싱글톤을 반환하고 없으면 한 번만 생성한다.
def get_retriever() -> LocalVectorRetriever:
    global _retriever
    if _retriever is None:
        logger.info("[Init] Build retriever")
        add_trace("init", "Build retriever")
        _retriever = build_retriever()
        add_trace("init", "Retriever ready", chunk_count=len(_retriever.chunks))
    return _retriever


# 공용 LLM 싱글톤을 반환하고 없으면 한 번만 생성한다.
def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        logger.info("[Init] Create chat model | model=%s", OPENAI_LLM_MODEL)
        add_trace("init", "Create chat model", model=OPENAI_LLM_MODEL)
        _llm = ChatOpenAI(model=OPENAI_LLM_MODEL, temperature=0)
    return _llm
