from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from app.core.config import logger
from app.prompt.common import REWRITE_QUERY_PROMPT
from app.prompt.crag.correction import (
    CRAG_GENERATE_ANSWER_PROMPT,
    CRAG_REGENERATE_ANSWER_PROMPT,
    CRAG_REVISE_ANSWER_PROMPT,
)
from app.schemas.rag import CRAGGraphState
from app.services.shared.dependencies import get_llm
from app.services.tracing import add_trace
from app.utils.crag import format_chat_history, format_documents, format_refined_evidence


# 정제된 evidence set을 바탕으로 1차 답변 초안을 생성한다.
def generate_answer(state: CRAGGraphState):
    logger.info(
        "CRAG node  | generate_answer | question=%r | evidence_count=%d",
        state["question"],
        state["evidence_count"],
    )
    add_trace(
        "crag_generate",
        "Generate CRAG draft answer from refined evidence",
        question=state["question"],
        evidence_count=state["evidence_count"],
        refine_quality=state["refine_quality"],
    )
    prompt = ChatPromptTemplate.from_template(CRAG_GENERATE_ANSWER_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
    generation = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "retrieval_summary": state["retrieval_reason"],
            "refine_summary": state["refine_summary"],
            "context": format_refined_evidence(state["refined_evidence"]),
        }
    )
    return {"generation": generation}


# 답변 평가 결과를 반영해 evidence 기준으로 초안을 다시 생성한다.
def regenerate_answer(state: CRAGGraphState):
    logger.info(
        "CRAG node  | regenerate_answer | question=%r | correction_retry_count=%d",
        state["question"],
        state["correction_retry_count"],
    )
    add_trace(
        "crag_regenerate",
        "Regenerate answer with refined evidence",
        missing_points=state["answer_missing_points"],
        unsupported_claims=state["answer_unsupported_claims"],
        correction_retry_count=state["correction_retry_count"],
    )
    prompt = ChatPromptTemplate.from_template(CRAG_REGENERATE_ANSWER_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
    generation = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "context": format_refined_evidence(state["refined_evidence"]),
            "generation": state["generation"],
            "missing_points": "\n".join(state["answer_missing_points"]) or "[없음]",
            "unsupported_claims": "\n".join(state["answer_unsupported_claims"]) or "[없음]",
        }
    )
    return {
        "generation": generation,
        "correction_retry_count": state["correction_retry_count"] + 1,
    }


# 로컬 검색을 한 번 더 시도할 가치가 있을 때 현재 질의를 더 구체적으로 재작성한다.
def rewrite_query(state: CRAGGraphState):
    logger.info(
        "CRAG node  | rewrite_query | previous_query=%r | retry_count=%d",
        state["current_query"],
        state["retry_count"],
    )
    add_trace(
        "crag_rewrite",
        "Rewrite CRAG retrieval query",
        previous_query=state["current_query"],
        retry_count=state["retry_count"],
    )
    suggested_query = state["rewritten_query"].strip()
    if suggested_query:
        return {
            "current_query": suggested_query,
            "retry_count": state["retry_count"] + 1,
        }

    prompt = ChatPromptTemplate.from_template(REWRITE_QUERY_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
    better_query = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "current_query": state["current_query"],
        }
    ).strip()
    return {
        "current_query": better_query,
        "retry_count": state["retry_count"] + 1,
    }


# 최신성 보강이 필요할 때 웹 검색 결과를 문자열로 반환한다.
@tool
def web_search(query: str) -> str:
    """최신성 보강이 필요할 때 웹 검색 결과를 문자열로 반환한다."""
    logger.warning("CRAG web search | query=%r", query)
    return GoogleSerperAPIWrapper().run(query)


# 외부 웹 검색을 실행하고 이후 검색 평가 단계가 다시 사용할 문서로 저장한다.
def web_search_node(state: CRAGGraphState):
    logger.warning(
        "CRAG node  | web_search_node | query=%r | retry_count=%d",
        state["current_query"],
        state["retry_count"],
    )
    add_trace(
        "crag_web_search",
        "Run CRAG web search",
        query=state["current_query"],
        retry_count=state["retry_count"],
    )
    search_result = web_search.invoke(state["current_query"])
    return {
        "documents": [
            Document(
                page_content=search_result,
                metadata={"source": "web_search"},
            )
        ],
        "web_search_used": True,
    }


# 최종 승인 전에 위험한 문장을 제거하고 evidence 기준으로 답변을 보수화한다.
def revise_answer(state: CRAGGraphState):
    logger.info(
        "CRAG node  | revise_answer | question=%r | final_revision_count=%d",
        state["question"],
        state["final_revision_count"],
    )
    add_trace(
        "crag_revise",
        "Revise answer conservatively from refined evidence",
        unsupported_claims=state["answer_unsupported_claims"],
        missing_points=state["answer_missing_points"],
        final_revision_count=state["final_revision_count"],
    )
    prompt = ChatPromptTemplate.from_template(CRAG_REVISE_ANSWER_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
    generation = chain.invoke(
        {
            "question": state["question"],
            "context": format_refined_evidence(state["refined_evidence"]),
            "generation": state["generation"],
            "unsupported_claims": "\n".join(state["answer_unsupported_claims"]) or "[없음]",
            "missing_points": "\n".join(state["answer_missing_points"]) or "[없음]",
        }
    )
    return {
        "generation": generation,
        "final_revision_count": state["final_revision_count"] + 1,
    }
