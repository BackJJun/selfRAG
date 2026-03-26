from langchain_core.documents import Document

from app.core.config import STREAM_CHUNK_SIZE
from app.schemas.rag import ChatTurn, GraphState


# 최근 대화 이력을 LLM 프롬프트에 넣기 쉬운 문자열로 정리한다.
def format_chat_history(chat_history: list[ChatTurn]) -> str:
    if not chat_history:
        return "[No prior chat history]"

    recent_turns = chat_history[-6:]
    lines = [f"{turn['role']}: {turn['content']}" for turn in recent_turns]
    return "\n".join(lines)


# 검색 문서를 문서 번호와 source를 포함한 문자열로 직렬화한다.
def format_documents(documents: list[Document]) -> str:
    if not documents:
        return "[No documents retrieved]"

    formatted: list[str] = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Document {index} | source={source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


# 정제된 evidence set을 Self-RAG 답변 단계에서 재사용할 수 있게 문자열로 변환한다.
def format_refined_evidence(items: list[dict]) -> str:
    if not items:
        return "[No refined evidence available]"

    formatted: list[str] = []
    for index, item in enumerate(items, start=1):
        formatted.append(
            "\n".join(
                [
                    f"[Evidence {index}]",
                    f"doc_index={item.get('doc_index', '')}",
                    f"source={item.get('source', 'unknown')}",
                    f"claim={item.get('claim', '')}",
                    f"support_text={item.get('support_text', '')}",
                    f"relevance_score={item.get('relevance_score', 0)}",
                    f"confidence={item.get('confidence', 'low')}",
                ]
            )
        )
    return "\n\n".join(formatted)


# 문서 또는 정제된 evidence 중 현재 단계에서 사용할 컨텍스트를 선택한다.
def format_generation_context(state: GraphState | dict) -> str:
    refined_evidence = state.get("refined_evidence", [])
    if refined_evidence:
        return format_refined_evidence(refined_evidence)
    return format_documents(state.get("documents", []))


# Self-RAG 기본 입력 상태를 생성한다.
def make_inputs(question: str, chat_history: list[ChatTurn]) -> GraphState:
    return {
        "question": question,
        "current_query": question,
        "chat_history": chat_history,
        "documents": [],
        "generation": "",
        "reflection_decision": "",
        "rewritten_query": "",
        "retry_count": 0,
        "web_search_used": False,
        "reflection_grounded": False,
        "reflection_complete": False,
        "reflection_relevant": False,
        "reflection_fresh": True,
        "reflection_issue_source": "none",
        "reflection_rationale": "",
        "retrieval_assessment_summary": "",
        "retrieval_used_doc_indexes": [],
        "retrieval_discarded_doc_indexes": [],
        "retrieval_document_assessments": [],
        "refined_evidence": [],
        "refine_summary": "",
        "refine_quality": "low",
        "evidence_count": 0,
    }


# Self-RAG 그래프 결과를 API 응답용 payload로 변환한다.
def result_to_payload(result: GraphState) -> dict:
    documents = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "content": doc.page_content,
        }
        for doc in result.get("documents", [])
    ]
    return {
        "answer": result.get("generation", ""),
        "current_query": result.get("current_query", ""),
        "retry_count": result.get("retry_count", 0),
        "reflection_decision": result.get("reflection_decision", ""),
        "reflection": {
            "grounded": result.get("reflection_grounded", False),
            "complete": result.get("reflection_complete", False),
            "relevant": result.get("reflection_relevant", False),
            "fresh": result.get("reflection_fresh", True),
            "issue_source": result.get("reflection_issue_source", "none"),
            "rationale": result.get("reflection_rationale", ""),
        },
        "retrieval": {
            "summary": result.get("retrieval_assessment_summary", ""),
            "used_doc_indexes": result.get("retrieval_used_doc_indexes", []),
            "discarded_doc_indexes": result.get("retrieval_discarded_doc_indexes", []),
            "documents": result.get("retrieval_document_assessments", []),
        },
        "documents": documents,
        "trace": result.get("trace", []),
    }


# 스트리밍 응답을 위해 긴 문자열을 일정 길이 청크로 자른다.
def chunk_text(text: str, size: int = STREAM_CHUNK_SIZE) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + size] for index in range(0, len(text), size)]
