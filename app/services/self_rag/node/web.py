from typing import Any

from langchain_core.documents import Document

from app.core.config import logger
from app.schemas.rag import GraphState
from app.services.shared.web_search import run_web_search
from app.services.tracing import add_trace


# 최신성 부족 시 공용 웹 검색 결과를 문서 형태로 감싸 다음 단계에 전달한다.
def web_search_node(state: GraphState) -> dict[str, Any]:
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
    search_result = run_web_search(state["current_query"])
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
        "retrieval_assessment_summary": "웹 검색 결과를 단일 문서 컨텍스트로 사용합니다.",
        "retrieval_used_doc_indexes": [1],
        "retrieval_discarded_doc_indexes": [],
        "retrieval_document_assessments": [
            {
                "doc_index": 1,
                "source": "web_search",
                "relevance_score": 5,
                "use": True,
                "rationale": "웹 검색으로 확보한 최신 결과를 그대로 사용합니다.",
            }
        ],
    }
