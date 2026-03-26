from app.prompt.self_rag.generation import GENERATE_ANSWER_PROMPT
from app.prompt.self_rag.reflection import REFLECT_ON_ANSWER_PROMPT
from app.prompt.self_rag.retrieval import EVALUATE_RETRIEVED_DOCUMENTS_PROMPT
from app.prompt.self_rag.revision import REVISE_ANSWER_PROMPT

__all__ = [
    "EVALUATE_RETRIEVED_DOCUMENTS_PROMPT",
    "GENERATE_ANSWER_PROMPT",
    "REFLECT_ON_ANSWER_PROMPT",
    "REVISE_ANSWER_PROMPT",
]
