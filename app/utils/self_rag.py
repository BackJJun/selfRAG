from langchain_core.documents import Document

from app.core.config import STREAM_CHUNK_SIZE
from app.schemas.chat import ChatTurn, GraphState


def format_chat_history(chat_history: list[ChatTurn]) -> str:
    if not chat_history:
        return "[No prior chat history]"

    recent_turns = chat_history[-6:]
    lines = [f"{turn['role']}: {turn['content']}" for turn in recent_turns]
    return "\n".join(lines)


def format_documents(documents: list[Document]) -> str:
    if not documents:
        return "[No documents retrieved]"

    formatted: list[str] = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Document {index} | source={source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


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
    }


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


def chunk_text(text: str, size: int = STREAM_CHUNK_SIZE) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + size] for index in range(0, len(text), size)]
