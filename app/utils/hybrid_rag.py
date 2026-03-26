from app.schemas.rag import HybridGraphState, ChatTurn


# 하이브리드 RAG 실행에 필요한 초기 상태를 생성한다.
def make_inputs(question: str, chat_history: list[ChatTurn]) -> HybridGraphState:
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
    }


# 하이브리드 RAG 결과를 API 응답용 payload로 변환한다.
def result_to_payload(result: HybridGraphState) -> dict:
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
        "retrieval": {
            "quality": result.get("retrieval_quality", ""),
            "score": result.get("retrieval_score", 0),
            "issue_type": result.get("retrieval_issue_type", "none"),
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
        "reflection": {
            "grounded": result.get("reflection_grounded", False),
            "complete": result.get("reflection_complete", False),
            "relevant": result.get("reflection_relevant", False),
            "fresh": result.get("reflection_fresh", True),
            "issue_source": result.get("reflection_issue_source", "none"),
            "rationale": result.get("reflection_rationale", ""),
        },
        "documents": documents,
        "trace": result.get("trace", []),
    }
