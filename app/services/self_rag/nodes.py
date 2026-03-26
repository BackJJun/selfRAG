from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from app.core.config import MAX_RETRIES, logger
from app.prompt.common import REWRITE_QUERY_PROMPT
from app.prompt.self_rag.self_rag_prompt import (
    GENERATE_ANSWER_PROMPT,
    REFLECT_ON_ANSWER_PROMPT,
    REVISE_ANSWER_PROMPT,
)
from app.schemas.chat import GraphState, ReflectionResult
from app.services.self_rag.dependencies import get_llm, get_retriever
from app.services.self_rag.tracing import add_trace
from app.services.self_rag.utils import format_chat_history, format_documents


@tool
def web_search(query: str) -> str:
    """Fallback web search for fresher or broader evidence."""
    logger.warning("Web search fallback triggered | query=%r", query)
    return GoogleSerperAPIWrapper().run(query)


def retrieve(state: GraphState):
    logger.info("Node start | retrieve | current_query=%r", state["current_query"])
    add_trace("retrieve", "Start retrieval", query=state["current_query"])
    documents = get_retriever().invoke(state["current_query"])
    logger.info("Node end   | retrieve | documents=%d", len(documents))
    add_trace(
        "retrieve",
        "Retrieved documents",
        document_count=len(documents),
        sources=[doc.metadata.get("source", "unknown") for doc in documents],
    )
    return {"documents": documents}


def generate_answer(state: GraphState):
    logger.info(
        "Node start | generate_answer | question=%r | history_turns=%d | documents=%d",
        state["question"],
        len(state["chat_history"]),
        len(state["documents"]),
    )
    add_trace(
        "generate",
        "Generate answer draft",
        question=state["question"],
        history_turns=len(state["chat_history"]),
        document_count=len(state["documents"]),
    )
    prompt = ChatPromptTemplate.from_template(GENERATE_ANSWER_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
    generation = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "context": format_documents(state["documents"]),
        }
    )
    logger.info("Node end   | generate_answer | answer_chars=%d", len(generation))
    add_trace("generate", "Draft answer ready", answer_chars=len(generation))
    return {"generation": generation}


def reflect_on_answer(state: GraphState):
    logger.info(
        "Node start | reflect_on_answer | current_query=%r | retry_count=%d",
        state["current_query"],
        state["retry_count"],
    )
    add_trace(
        "reflect",
        "Evaluate draft answer",
        current_query=state["current_query"],
        retry_count=state["retry_count"],
    )
    prompt = ChatPromptTemplate.from_template(REFLECT_ON_ANSWER_PROMPT)
    structured_llm = get_llm().with_structured_output(ReflectionResult)
    chain = prompt | structured_llm
    result = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "current_query": state["current_query"],
            "context": format_documents(state["documents"]),
            "generation": state["generation"],
        }
    )

    logger.info(
        "Node end   | reflect_on_answer | decision=%s | grounded=%s | complete=%s | relevant=%s | fresh=%s | issue_source=%s | rewritten_query=%r",
        result.decision,
        result.grounded,
        result.complete,
        result.relevant,
        result.fresh,
        result.issue_source,
        (result.rewritten_query or "").strip(),
    )
    add_trace(
        "reflect",
        "Reflection complete",
        decision=result.decision,
        grounded=result.grounded,
        complete=result.complete,
        relevant=result.relevant,
        fresh=result.fresh,
        issue_source=result.issue_source,
        rationale=result.rationale,
        rewritten_query=(result.rewritten_query or "").strip(),
    )
    return {
        "reflection_decision": result.decision,
        "rewritten_query": (result.rewritten_query or "").strip(),
        "reflection_grounded": result.grounded,
        "reflection_complete": result.complete,
        "reflection_relevant": result.relevant,
        "reflection_fresh": result.fresh,
        "reflection_issue_source": result.issue_source,
        "reflection_rationale": result.rationale,
    }


def rewrite_query(state: GraphState):
    logger.info(
        "Node start | rewrite_query | previous_query=%r | retry_count=%d",
        state["current_query"],
        state["retry_count"],
    )
    add_trace(
        "rewrite",
        "Rewrite retrieval query",
        previous_query=state["current_query"],
        retry_count=state["retry_count"],
    )
    suggested_query = state["rewritten_query"].strip()
    if suggested_query:
        logger.info("Node end   | rewrite_query | using_reflection_query=%r", suggested_query)
        add_trace("rewrite", "Use reflection query", rewritten_query=suggested_query)
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

    logger.info("Node end   | rewrite_query | generated_query=%r", better_query)
    add_trace("rewrite", "Generated new query", rewritten_query=better_query)
    return {
        "current_query": better_query,
        "retry_count": state["retry_count"] + 1,
    }


def revise_answer(state: GraphState):
    logger.info(
        "Node start | revise_answer | question=%r | documents=%d",
        state["question"],
        len(state["documents"]),
    )
    add_trace(
        "revise",
        "Revise final answer after exhausted retrieval",
        question=state["question"],
        document_count=len(state["documents"]),
        issue_source=state["reflection_issue_source"],
    )
    prompt = ChatPromptTemplate.from_template(REVISE_ANSWER_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
    revised_answer = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "context": format_documents(state["documents"]),
            "generation": state["generation"],
        }
    )
    logger.info("Node end   | revise_answer | answer_chars=%d", len(revised_answer))
    add_trace("revise", "Revised final answer ready", answer_chars=len(revised_answer))
    return {
        "generation": revised_answer,
        "reflection_decision": "answer",
    }


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
    }


def route_after_reflection(state: GraphState):
    logger.info(
        "Router     | route_after_reflection | decision=%s | retry_count=%d | web_search_used=%s | issue_source=%s | fresh=%s",
        state["reflection_decision"],
        state["retry_count"],
        state["web_search_used"],
        state["reflection_issue_source"],
        state["reflection_fresh"],
    )

    if state["reflection_decision"] == "answer":
        logger.info("Router     | route_after_reflection | next=end")
        add_trace("route", "Finish with current answer", next_step="end")
        return "end"

    if state["web_search_used"]:
        logger.info("Router     | route_after_reflection | next=revise_answer")
        add_trace(
            "route",
            "Revise answer after web-search fallback instead of looping again",
            next_step="revise_answer",
            reason="web_search_already_used",
        )
        return "revise_answer"

    if (not state["reflection_fresh"]) or state["reflection_issue_source"] == "freshness_problem":
        logger.info("Router     | route_after_reflection | next=web_search_node")
        add_trace(
            "route",
            "Go to web search because freshness is insufficient",
            next_step="web_search_node",
        )
        return "web_search_node"

    if state["reflection_issue_source"] == "answer_problem":
        logger.info("Router     | route_after_reflection | next=revise_answer")
        add_trace(
            "route",
            "Revise answer because retrieval looks usable but synthesis is weak",
            next_step="revise_answer",
        )
        return "revise_answer"

    if state["retry_count"] < MAX_RETRIES and state["reflection_issue_source"] in {
        "query_problem",
        "retrieval_problem",
        "mixed",
        "none",
    }:
        logger.info("Router     | route_after_reflection | next=rewrite_query")
        add_trace(
            "route",
            "Retry retrieval with rewritten query",
            next_step="rewrite_query",
            issue_source=state["reflection_issue_source"],
        )
        return "rewrite_query"

    logger.info("Router     | route_after_reflection | next=web_search_node")
    add_trace("route", "Fallback to web search", next_step="web_search_node")
    return "web_search_node"
