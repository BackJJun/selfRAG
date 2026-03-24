import argparse
import asyncio
import json
import logging
import math
import os
import secrets
import threading
import time
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal, TypedDict
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
LOG_LEVEL = os.getenv("SELF_RAG_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("self_rag")

logger.info("=" * 80)
logger.info("LangGraph Self-RAG-lite Demo for Python 3.11")
logger.info("=" * 80)


# Python 3.11 기준으로 작성한 대화 이력 타입이다.
class ChatTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class GraphState(TypedDict):
    question: str
    current_query: str
    chat_history: list[ChatTurn]
    documents: list[Document]
    generation: str
    reflection_decision: str
    rewritten_query: str
    retry_count: int


class ReflectionResult(BaseModel):
    decision: Literal["answer", "retrieve_more"] = Field(
        description="Return answer if grounded enough, otherwise retrieve_more."
    )
    rationale: str = Field(
        description="Why the current answer is sufficiently supported or not."
    )
    rewritten_query: str = Field(
        description="Better retrieval query for the next pass. Empty if not needed."
    )


DEFAULT_PDF_PATH = BASE_DIR / "data" / "SPRi AI Brief_10월호_산업동향_1002_F.pdf"
DEFAULT_TEXT_PATH = BASE_DIR / "data" / "Sample_Text.txt"
FILE_PATH = Path(os.getenv("SELF_RAG_SOURCE_PATH", str(DEFAULT_PDF_PATH)))
CHUNK_SIZE = int(os.getenv("SELF_RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("SELF_RAG_CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("SELF_RAG_TOP_K", "3"))
MAX_RETRIES = int(os.getenv("SELF_RAG_MAX_RETRIES", "2"))
STREAM_CHUNK_SIZE = int(os.getenv("SELF_RAG_STREAM_CHUNK_SIZE", "18"))
SESSION_COOKIE_NAME = "self_rag_session"


def load_source_documents() -> list[Document]:
    """
    로컬 문서를 읽는다.

    우선순위:
    1. SELF_RAG_SOURCE_PATH
    2. 기본 PDF
    3. 기본 TXT
    """
    candidate_paths = [FILE_PATH, DEFAULT_PDF_PATH, DEFAULT_TEXT_PATH]

    for path in candidate_paths:
        if not path.exists():
            logger.debug("Source path not found: %s", path)
            continue

        if path.suffix.lower() == ".pdf":
            logger.info("Loading PDF source: %s", path)
            return PyPDFLoader(str(path)).load()

        if path.suffix.lower() == ".txt":
            logger.info("Loading text source: %s", path)
            text = path.read_text(encoding="utf-8")
            return [Document(page_content=text, metadata={"source": str(path)})]

    searched = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"No source document found. Checked: {searched}")


class LocalVectorRetriever:
    """
    외부 DB 없이 메모리 안에서만 동작하는 간단한 retriever.

    문서를 chunk로 나눈 뒤 임베딩해 두고,
    질의 임베딩과 cosine similarity를 직접 계산한다.
    """

    def __init__(self, documents: list[Document], embedding_model: OpenAIEmbeddings):
        self.embedding_model = embedding_model
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.chunks = splitter.split_documents(documents)
        chunk_texts = [chunk.page_content for chunk in self.chunks]
        logger.info("Embedding %d chunks for in-memory retriever", len(chunk_texts))
        self.chunk_embeddings = self.embedding_model.embed_documents(chunk_texts)
        logger.info("Prepared %d chunks for in-memory retrieval", len(self.chunks))

    @staticmethod
    def cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def invoke(self, query: str) -> list[Document]:
        logger.info("Retriever invoked | query=%r", query)
        query_embedding = self.embedding_model.embed_query(query)
        scored_chunks: list[tuple[float, Document]] = []

        for chunk, embedding in zip(self.chunks, self.chunk_embeddings):
            score = self.cosine_similarity(query_embedding, embedding)
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        top_scored = scored_chunks[:TOP_K]
        logger.info(
            "Retriever finished | top_k=%d | scores=%s",
            TOP_K,
            [round(score, 4) for score, _ in top_scored],
        )
        return [chunk for _, chunk in top_scored]


def build_retriever() -> LocalVectorRetriever:
    documents = load_source_documents()
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return LocalVectorRetriever(documents, embeddings_model)


logger.info("[Part 1] Build retriever and tools")
logger.info("-" * 80)
retriever = build_retriever()


@tool
def web_search(query: str) -> str:
    """로컬 문서만으로 부족할 때 사용하는 최신 정보 검색 도구."""
    logger.warning("Web search fallback triggered | query=%r", query)
    return GoogleSerperAPIWrapper().run(query)


llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)


def format_chat_history(chat_history: list[ChatTurn]) -> str:
    """
    최근 대화 이력을 prompt용 문자열로 정리한다.

    이력이 너무 길어지면 비용만 커지므로 최근 6개 turn만 사용한다.
    """
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


def retrieve(state: GraphState):
    logger.info("Node start | retrieve | current_query=%r", state["current_query"])
    documents = retriever.invoke(state["current_query"])
    logger.info("Node end   | retrieve | documents=%d", len(documents))
    return {"documents": documents}


def generate_answer(state: GraphState):
    """
    현재 검색 결과와 대화 이력을 바탕으로 답변 초안을 만든다.
    """
    logger.info(
        "Node start | generate_answer | question=%r | history_turns=%d | documents=%d",
        state["question"],
        len(state["chat_history"]),
        len(state["documents"]),
    )
    prompt = ChatPromptTemplate.from_template(
        """
        You are answering the user's question using retrieved evidence.

        Rules:
        - Use the retrieved context first.
        - Consider the recent chat history for follow-up questions.
        - If evidence is incomplete, say what is uncertain.
        - Do not invent unsupported facts.
        - Write the answer in Korean.

        Recent chat history:
        {chat_history}

        Question:
        {question}

        Context:
        {context}
        """
    )

    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "context": format_documents(state["documents"]),
        }
    )
    logger.info("Node end   | generate_answer | answer_chars=%d", len(generation))
    return {"generation": generation}


def reflect_on_answer(state: GraphState):
    """
    Self-RAG 핵심 단계.

    초안 답변이 충분히 grounded 되어 있는지 확인하고,
    부족하면 다음 retrieval에 쓸 더 나은 query를 만든다.
    """
    logger.info(
        "Node start | reflect_on_answer | current_query=%r | retry_count=%d",
        state["current_query"],
        state["retry_count"],
    )
    prompt = ChatPromptTemplate.from_template(
        """
        You are a self-reflection module for a retrieval-augmented assistant.

        Review:
        - recent chat history
        - the user question
        - current retrieval query
        - retrieved context
        - draft answer

        Return:
        - decision="answer" if the draft is sufficiently supported.
        - decision="retrieve_more" if the draft is weak, unsupported, incomplete,
          or likely needs broader/fresher evidence.

        If decision is retrieve_more, provide a rewritten_query.

        Recent chat history:
        {chat_history}

        Question:
        {question}

        Current query:
        {current_query}

        Retrieved context:
        {context}

        Draft answer:
        {generation}
        """
    )

    structured_llm = llm.with_structured_output(ReflectionResult)
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
    return {
        "reflection_decision": result.decision,
        "rewritten_query": (result.rewritten_query or "").strip(),
    }


def rewrite_query(state: GraphState):
    """
    reflection에서 만든 query가 있으면 그대로 쓰고,
    없으면 별도 prompt로 다시 쓴다.
    """
    logger.info(
        "Node start | rewrite_query | previous_query=%r | retry_count=%d",
        state["current_query"],
        state["retry_count"],
    )
    suggested_query = state["rewritten_query"].strip()
    if suggested_query:
        logger.info("Node end   | rewrite_query | using_reflection_query=%r", suggested_query)
        return {
            "current_query": suggested_query,
            "retry_count": state["retry_count"] + 1,
        }

    prompt = ChatPromptTemplate.from_template(
        """
        Rewrite the user's question into a better retrieval query.

        Requirements:
        - Keep it concise.
        - Preserve the original intent.
        - Reflect the recent chat context if it matters.
        - Add useful search keywords.
        - Return only the rewritten query.

        Recent chat history:
        {chat_history}

        Original question:
        {question}

        Current query:
        {current_query}
        """
    )

    chain = prompt | llm | StrOutputParser()
    better_query = chain.invoke(
        {
            "chat_history": format_chat_history(state["chat_history"]),
            "question": state["question"],
            "current_query": state["current_query"],
        }
    ).strip()

    logger.info("Node end   | rewrite_query | generated_query=%r", better_query)
    return {
        "current_query": better_query,
        "retry_count": state["retry_count"] + 1,
    }


def web_search_node(state: GraphState):
    """
    로컬 재검색 한도를 넘기면 웹 검색으로 넘어간다.
    """
    logger.warning(
        "Node start | web_search_node | query=%r | retry_count=%d",
        state["current_query"],
        state["retry_count"],
    )
    search_result = web_search.invoke(state["current_query"])
    logger.info("Node end   | web_search_node | result_chars=%d", len(search_result))
    return {
        "documents": [
            Document(
                page_content=search_result,
                metadata={"source": "web_search"},
            )
        ]
    }


def route_after_reflection(state: GraphState):
    logger.info(
        "Router     | route_after_reflection | decision=%s | retry_count=%d",
        state["reflection_decision"],
        state["retry_count"],
    )
    if state["reflection_decision"] == "answer":
        logger.info("Router     | route_after_reflection | next=end")
        return "end"
    if state["retry_count"] < MAX_RETRIES:
        logger.info("Router     | route_after_reflection | next=rewrite_query")
        return "rewrite_query"
    logger.info("Router     | route_after_reflection | next=web_search_node")
    return "web_search_node"


logger.info("[Part 2] Build Self-RAG graph")
logger.info("-" * 80)
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("reflect_on_answer", reflect_on_answer)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("web_search_node", web_search_node)
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
    },
)
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("web_search_node", "generate_answer")
app = workflow.compile()
logger.info("Self-RAG graph compiled")


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
    }


def run_self_rag(question: str, chat_history: list[ChatTurn]) -> GraphState:
    """
    동기 실행 경로.
    """
    started_at = time.perf_counter()
    logger.info(
        "Run start  | question=%r | history_turns=%d",
        question,
        len(chat_history),
    )
    result = app.invoke(make_inputs(question, chat_history))
    logger.info(
        "Run end    | question=%r | elapsed=%.2fs | decision=%s | retry_count=%d",
        question,
        time.perf_counter() - started_at,
        result.get("reflection_decision", ""),
        result.get("retry_count", 0),
    )
    return result


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
    }


def chunk_text(text: str, size: int) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + size] for index in range(0, len(text), size)]


CHAT_SESSIONS: dict[str, list[ChatTurn]] = {}
CHAT_LOCK = threading.Lock()


def get_or_create_session_id(handler: BaseHTTPRequestHandler) -> tuple[str, bool]:
    raw_cookie = handler.headers.get("Cookie", "")
    cookie = SimpleCookie()
    cookie.load(raw_cookie)

    session_id = cookie.get(SESSION_COOKIE_NAME)
    if session_id and session_id.value:
        logger.debug("Session reuse | session_id=%s", session_id.value)
        return session_id.value, False

    new_session_id = secrets.token_hex(16)
    logger.info("Session create | session_id=%s", new_session_id)
    return new_session_id, True


def get_session_history(session_id: str) -> list[ChatTurn]:
    with CHAT_LOCK:
        history = CHAT_SESSIONS.setdefault(session_id, [])
        logger.debug("Session read  | session_id=%s | history_turns=%d", session_id, len(history))
        return [turn.copy() for turn in history]


def append_session_history(session_id: str, user_message: str, assistant_message: str) -> None:
    with CHAT_LOCK:
        history = CHAT_SESSIONS.setdefault(session_id, [])
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})
        logger.info(
            "Session write | session_id=%s | history_turns=%d",
            session_id,
            len(history),
        )


jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


class SelfRAGWebHandler(BaseHTTPRequestHandler):
    """
    GET /         : 메인 화면
    GET /stream   : SSE 스트리밍 응답

    화면 자체는 Jinja2로 렌더링하고,
    답변은 SSE로 chunk 단위 전송한다.
    """

    def render_page(
        self,
        session_id: str,
        set_cookie: bool,
        error: str = "",
    ) -> None:
        logger.info("HTTP render  | path=/ | session_id=%s", session_id)
        template = jinja_env.get_template("index.html")
        history = get_session_history(session_id)

        html = template.render(
            error=error,
            history=history,
        )
        body = html.encode("utf-8")

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if set_cookie:
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE_NAME}={session_id}; Path=/; HttpOnly; SameSite=Lax",
            )
        self.end_headers()
        self.wfile.write(body)

    def send_sse(self, event: str, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        message = f"event: {event}\ndata: {payload}\n\n".encode("utf-8")
        self.wfile.write(message)
        self.wfile.flush()
        logger.debug("SSE send     | event=%s | keys=%s", event, list(data.keys()))

    def handle_stream(self) -> None:
        session_id, set_cookie = get_or_create_session_id(self)
        query = parse_qs(urlparse(self.path).query)
        question = query.get("question", [""])[0].strip()
        logger.info(
            "HTTP stream  | path=/stream | session_id=%s | question=%r",
            session_id,
            question,
        )

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        if set_cookie:
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE_NAME}={session_id}; Path=/; HttpOnly; SameSite=Lax",
            )
        self.end_headers()

        if not question:
            logger.warning("HTTP stream  | empty question | session_id=%s", session_id)
            self.send_sse("error", {"message": "질문을 입력해 주세요."})
            return

        try:
            history = get_session_history(session_id)
            self.send_sse("status", {"message": "검색과 답변 생성을 시작합니다."})

            result = run_self_rag(question, history)
            payload = result_to_payload(result)

            self.send_sse("status", {"message": "답변을 스트리밍합니다."})
            for piece in chunk_text(payload["answer"], STREAM_CHUNK_SIZE):
                self.send_sse("token", {"text": piece})
                time.sleep(0.02)

            append_session_history(session_id, question, payload["answer"])
            self.send_sse(
                "result",
                {
                    "current_query": payload["current_query"],
                    "retry_count": payload["retry_count"],
                    "reflection_decision": payload["reflection_decision"],
                    "documents": payload["documents"],
                },
            )
            self.send_sse("done", {"ok": True})
            logger.info(
                "HTTP stream  | complete | session_id=%s | answer_chars=%d",
                session_id,
                len(payload["answer"]),
            )
        except Exception as exc:
            logger.exception("HTTP stream  | error | session_id=%s", session_id)
            self.send_sse("error", {"message": str(exc)})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        logger.info("HTTP GET     | path=%s", parsed.path)
        if parsed.path == "/stream":
            self.handle_stream()
            return

        session_id, set_cookie = get_or_create_session_id(self)
        self.render_page(session_id=session_id, set_cookie=set_cookie)

    def do_POST(self) -> None:
        logger.warning("HTTP POST    | path=%s | method_not_allowed", self.path)
        self.send_error(HTTPStatus.METHOD_NOT_ALLOWED, "Use GET or /stream SSE endpoint.")

    def log_message(self, format: str, *args) -> None:
        logger.info("HTTP access  | %s - %s", self.address_string(), format % args)


def serve_web(host: str = "127.0.0.1", port: int = 8000) -> None:
    logger.info(
        "Server init | host=%s | port=%d | source_path=%s | top_k=%d | max_retries=%d | stream_chunk_size=%d",
        host,
        port,
        FILE_PATH,
        TOP_K,
        MAX_RETRIES,
        STREAM_CHUNK_SIZE,
    )
    server = ThreadingHTTPServer((host, port), SelfRAGWebHandler)
    logger.info("Server ready | url=http://%s:%d", host, port)
    server.serve_forever()


async def cli_main() -> None:
    logger.info("=" * 80)
    logger.info("Self-RAG CLI started")
    logger.info("Type 'quit' or 'exit' to stop")
    logger.info("=" * 80)

    history: list[ChatTurn] = []

    while True:
        user_question = input("\nYou: ").strip()
        if user_question.lower() in ["quit", "exit", "종료"]:
            logger.info("CLI session ended by user")
            break
        if not user_question:
            continue

        try:
            result = await asyncio.to_thread(run_self_rag, user_question, history)
            answer = result.get("generation", "")
            history.append({"role": "user", "content": user_question})
            history.append({"role": "assistant", "content": answer})
            logger.info("CLI answer   | answer_chars=%d | history_turns=%d", len(answer), len(history))
            print(f"\nAI: {answer}")
            print("-" * 80)
        except Exception as exc:
            logger.exception("CLI runtime error")
            print(f"\n[Runtime Error] {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Self-RAG demo for Python 3.11")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "web":
        serve_web(host=args.host, port=args.port)
    else:
        asyncio.run(cli_main())
