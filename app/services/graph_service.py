import threading
import time
from contextvars import ContextVar
from typing import Any

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.core.config import LANGSMITH_PROJECT, MAX_RETRIES, OPENAI_LLM_MODEL, STREAM_CHUNK_SIZE, logger
from app.prompt.common import REWRITE_QUERY_PROMPT
from app.prompt.self_rag.self_rag_prompt import (
    GENERATE_ANSWER_PROMPT,
    REFLECT_ON_ANSWER_PROMPT,
    REVISE_ANSWER_PROMPT,
)
from app.schemas.chat import ChatTurn, GraphState, ReflectionResult
from app.services.retriever import LocalVectorRetriever, build_retriever


_graph_lock = threading.Lock()
_graph_app: Any | None = None
_retriever: LocalVectorRetriever | None = None
_llm: ChatOpenAI | None = None
_trace_events_var: ContextVar[list[dict[str, Any]] | None] = ContextVar("trace_events", default=None)
_trace_started_at_var: ContextVar[float | None] = ContextVar("trace_started_at", default=None)


@tool
# ? ?? ?? ??? ?? ??.
def web_search(query: str) -> str:
    """Fallback web search for fresher or broader evidence."""
    logger.warning("Web search fallback triggered | query=%r", query)
    return GoogleSerperAPIWrapper().run(query)


# ?? ?? ??? trace ???? ???? ????.
def add_trace(stage: str, message: str, **details: Any) -> None:
    events = _trace_events_var.get()
    started_at = _trace_started_at_var.get()
    if events is None or started_at is None:
        return

    events.append(
        {
            "stage": stage,
            "message": message,
            "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
            "details": details,
        }
    )


# ?? retriever? 1?? ???? ?????.
def get_retriever() -> LocalVectorRetriever:
    global _retriever
    if _retriever is None:
        logger.info("[Init] Build retriever")
        add_trace("init", "Build retriever")
        _retriever = build_retriever()
        add_trace("init", "Retriever ready", chunk_count=len(_retriever.chunks))
    return _retriever


# ?? LLM ?????? 1?? ???? ?????.
def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        logger.info("[Init] Create chat model | model=%s", OPENAI_LLM_MODEL)
        add_trace("init", "Create chat model", model=OPENAI_LLM_MODEL)
        _llm = ChatOpenAI(model=OPENAI_LLM_MODEL, temperature=0)
    return _llm


# ?? ?? ??? ????? ?? ?? ???? ????.
def format_chat_history(chat_history: list[ChatTurn]) -> str:
    if not chat_history:
        return "[No prior chat history]"

    recent_turns = chat_history[-6:]
    lines = [f"{turn['role']}: {turn['content']}" for turn in recent_turns]
    return "\n".join(lines)


# ??? ?? ??? ???? ????? ???? ????.
def format_documents(documents: list[Document]) -> str:
    if not documents:
        return "[No documents retrieved]"

    formatted: list[str] = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Document {index} | source={source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


# ?? ??? ?? retriever? ??? ?? ??? ????.
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


# ?? ??? ???? ????? ??? ?? ??? ????.
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


# ??? ??? ???? ???? ?? ?? ?? ??? ????.
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
        "Node end   | reflect_on_answer | decision=%s | rewritten_query=%r",
        result.decision,
        (result.rewritten_query or "").strip(),
    )
    add_trace(
        "reflect",
        "Reflection complete",
        decision=result.decision,
        rewritten_query=(result.rewritten_query or "").strip(),
    )
    return {
        "reflection_decision": result.decision,
        "rewritten_query": (result.rewritten_query or "").strip(),
    }


# ?? ??? ???? ?? ?? ??? ?????.
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


# ??? ?? ??? ??? ? ?? ??? ???? ?? ???? ?? ????.
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


# ??? ??? ??? ? ? ?? ??? ?? ??? ????.
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


# ?? ??? ??? ??? ?? ?? ??? ????.
def route_after_reflection(state: GraphState):
    logger.info(
        "Router     | route_after_reflection | decision=%s | retry_count=%d | web_search_used=%s",
        state["reflection_decision"],
        state["retry_count"],
        state["web_search_used"],
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
        )
        return "revise_answer"
    if state["retry_count"] < MAX_RETRIES:
        logger.info("Router     | route_after_reflection | next=rewrite_query")
        add_trace("route", "Retry retrieval with rewritten query", next_step="rewrite_query")
        return "rewrite_query"
    logger.info("Router     | route_after_reflection | next=web_search_node")
    add_trace("route", "Fallback to web search", next_step="web_search_node")
    return "web_search_node"


# Self-RAG ?? ????? ???? ???? ?????.
def build_graph_app():
    logger.info("[Init] Compile Self-RAG graph")
    add_trace("init", "Compile Self-RAG graph")
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("reflect_on_answer", reflect_on_answer)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("revise_answer", revise_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", "reflect_on_answer")
    workflow.add_conditional_edges(
        "reflect_on_answer",
        route_after_reflection,
        {
            "end": END,
            "rewrite_query": "rewrite_query",
            "web_search_node": "web_search_node",
            "revise_answer": "revise_answer",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_search_node", "generate_answer")
    workflow.add_edge("revise_answer", END)
    compiled = workflow.compile()
    logger.info("[Init] Self-RAG graph compiled")
    add_trace("init", "Graph ready")
    return compiled


# ?? ??? ?? ?? ???? ??? ?????.
def get_graph_app():
    global _graph_app
    if _graph_app is None:
        with _graph_lock:
            if _graph_app is None:
                _graph_app = build_graph_app()
    return _graph_app


# ?? ?? ?? LLM, retriever, graph? ?? ????.
def warmup_graph_dependencies() -> None:
    with _graph_lock:
        get_llm()
        get_retriever()
        global _graph_app
        if _graph_app is None:
            _graph_app = build_graph_app()


# ??? ??? ??? ?? ???? ???.
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
    }


# Self-RAG ???? ??? ???? trace ??? ?? ????.
def run_self_rag(question: str, chat_history: list[ChatTurn]) -> GraphState:
    started_at = time.perf_counter()
    trace_events: list[dict[str, Any]] = []
    trace_token = _trace_events_var.set(trace_events)
    started_token = _trace_started_at_var.set(started_at)

    try:
        logger.info(
            "Run start  | question=%r | history_turns=%d",
            question,
            len(chat_history),
        )
        add_trace("run", "Start Self-RAG run", question=question, history_turns=len(chat_history))
        invoke_config: RunnableConfig = {
            "run_name": "self_rag_graph",
            "tags": ["self-rag", LANGSMITH_PROJECT],
            "metadata": {
                "question": question,
                "history_turns": len(chat_history),
                "max_retries": MAX_RETRIES,
            },
        }
        result = get_graph_app().invoke(make_inputs(question, chat_history), config=invoke_config)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "Run end    | question=%r | elapsed=%.2fs | decision=%s | retry_count=%d",
            question,
            elapsed_ms / 1000,
            result.get("reflection_decision", ""),
            result.get("retry_count", 0),
        )
        add_trace(
            "run",
            "Self-RAG run completed",
            elapsed_ms=elapsed_ms,
            decision=result.get("reflection_decision", ""),
            retry_count=result.get("retry_count", 0),
        )
        result["trace"] = list(trace_events)
        return result
    finally:
        _trace_events_var.reset(trace_token)
        _trace_started_at_var.reset(started_token)


# ?? ??? ??? API ??? dict ??? ????.
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
        "documents": documents,
        "trace": result.get("trace", []),
    }


# ? ???? ???? ??? ?? ??? ???.
def chunk_text(text: str, size: int = STREAM_CHUNK_SIZE) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + size] for index in range(0, len(text), size)]
