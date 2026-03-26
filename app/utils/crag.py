from app.schemas.rag import CRAGGraphState, ChatTurn
from app.utils.self_rag import format_chat_history, format_documents


# CRAG 실행에 필요한 기본 상태를 만들어 그래프 입력으로 전달한다.
def make_inputs(question: str, chat_history: list[ChatTurn]) -> CRAGGraphState:
    return {
        "question": question,
        "current_query": question,
        "chat_history": chat_history,
        "documents": [],
        "generation": "",
        "rewritten_query": "",
        "retry_count": 0,
        "web_search_used": False,
        "retrieval_quality": "",
        "retrieval_score": 0,
        "retrieval_issue_type": "none",
        "retrieval_reason": "",
        "retrieval_should_retry": False,
        "retrieval_should_use_web": False,
        "retrieval_used_doc_indexes": [],
        "retrieval_discarded_doc_indexes": [],
        "retrieval_document_assessments": [],
        "refined_evidence": [],
        "refine_summary": "",
        "refine_quality": "low",
        "evidence_count": 0,
        "answer_grounded": False,
        "answer_complete": False,
        "answer_relevant": False,
        "answer_rationale": "",
        "answer_next_action": "",
        "answer_missing_points": [],
        "answer_unsupported_claims": [],
        "final_answer_approved": False,
        "final_answer_action": "",
        "final_answer_rationale": "",
        "correction_retry_count": 0,
        "final_revision_count": 0,
    }


# 정제된 evidence set을 LLM 프롬프트에 넣기 쉬운 텍스트로 변환한다.
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


# CRAG 최종 상태를 API나 CLI에서 다루기 쉬운 payload 형태로 변환한다.
def result_to_payload(result: CRAGGraphState) -> dict:
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
        "correction_retry_count": result.get("correction_retry_count", 0),
        "retrieval": {
            "summary": result.get("retrieval_reason", ""),
            "used_doc_indexes": result.get("retrieval_used_doc_indexes", []),
            "discarded_doc_indexes": result.get("retrieval_discarded_doc_indexes", []),
            "documents": result.get("retrieval_document_assessments", []),
        },
        "refine": {
            "summary": result.get("refine_summary", ""),
            "quality": result.get("refine_quality", "low"),
            "evidence_count": result.get("evidence_count", 0),
            "items": result.get("refined_evidence", []),
        },
        "answer_assessment": {
            "grounded": result.get("answer_grounded", False),
            "complete": result.get("answer_complete", False),
            "relevant": result.get("answer_relevant", False),
            "rationale": result.get("answer_rationale", ""),
            "next_action": result.get("answer_next_action", ""),
            "missing_points": result.get("answer_missing_points", []),
            "unsupported_claims": result.get("answer_unsupported_claims", []),
        },
        "final_assessment": {
            "approved": result.get("final_answer_approved", False),
            "action": result.get("final_answer_action", ""),
            "rationale": result.get("final_answer_rationale", ""),
        },
        "documents": documents,
        "trace": result.get("trace", []),
    }


__all__ = [
    "format_chat_history",
    "format_documents",
    "format_refined_evidence",
    "make_inputs",
    "result_to_payload",
]
