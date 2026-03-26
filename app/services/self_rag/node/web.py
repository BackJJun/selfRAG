from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.tools import tool

from app.core.config import logger
from app.schemas.chat import GraphState
from app.services.tracing import add_trace


@tool
def web_search(query: str) -> str:
    """Fallback web search for fresher or broader evidence."""
    logger.warning("Web search fallback triggered | query=%r", query)
    return GoogleSerperAPIWrapper().run(query)


def web_search_node(state: GraphState):
    logger.warning(
        "Node start | web_search_node | query=%r | retry_count=%d",
        state["current_query"],
        state["retry_count"],
    )
    add_trace(
        "web_search",
        "Fallback web search",
        query=state["current_query"],
        retry_count=state["retry_count"],
        issue_source=state["reflection_issue_source"],
    )
    search_result = web_search.invoke(state["current_query"])
    logger.info("Node end   | web_search_node | result_chars=%d", len(search_result))
    add_trace("web_search", "Web search result ready", result_chars=len(search_result))
    return {
        "documents": [
            Document(
                page_content=search_result,
                metadata={"source": "web_search"},
            )
        ],
        "web_search_used": True,
        "retrieval_assessment_summary": "웹 검색 결과를 단일 문서 컨텍스트로 사용했습니다.",
        "retrieval_used_doc_indexes": [1],
        "retrieval_discarded_doc_indexes": [],
        "retrieval_document_assessments": [
            {
                "doc_index": 1,
                "source": "web_search",
                "relevance_score": 5,
                "use": True,
                "rationale": "웹 검색 폴백 결과를 그대로 사용합니다.",
            }
        ],
    }
