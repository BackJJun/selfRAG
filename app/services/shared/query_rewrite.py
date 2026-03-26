from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.prompt.common import REWRITE_QUERY_PROMPT
from app.services.shared.dependencies import get_llm
from app.utils.self_rag import format_chat_history

_REWRITE_QUERY_TEMPLATE = ChatPromptTemplate.from_template(REWRITE_QUERY_PROMPT)


# 공통 질의 재작성 프롬프트를 사용해 다음 검색 질의를 생성한다.
def resolve_rewritten_query(
    *,
    question: str,
    chat_history: list[dict[str, Any]],
    current_query: str,
    suggested_query: str,
) -> str:
    if suggested_query.strip():
        return suggested_query.strip()

    chain = _REWRITE_QUERY_TEMPLATE | get_llm() | StrOutputParser()
    return chain.invoke(
        {
            "chat_history": format_chat_history(chat_history),
            "question": question,
            "current_query": current_query,
        }
    ).strip()
