import threading

from langchain_openai import ChatOpenAI

from app.services.retriever import LocalVectorRetriever
from app.services.self_rag.dependencies import get_llm as get_shared_llm
from app.services.self_rag.dependencies import get_retriever as get_shared_retriever

_graph_lock = threading.Lock()


# CRAG 그래프 초기화 시 동시 빌드를 막기 위한 전용 락을 반환한다.
def get_graph_lock() -> threading.Lock:
    return _graph_lock


# Self-RAG와 같은 리트리버 싱글톤을 재사용해 문서 인덱싱 비용을 줄인다.
def get_retriever() -> LocalVectorRetriever:
    return get_shared_retriever()


# Self-RAG와 같은 LLM 싱글톤을 재사용해 평가와 생성 설정을 일관되게 유지한다.
def get_llm() -> ChatOpenAI:
    return get_shared_llm()
