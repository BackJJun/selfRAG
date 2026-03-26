from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ChatTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class GraphState(TypedDict):
    question: str
    current_query: str
    chat_history: list[ChatTurn]
    documents: list
    generation: str
    reflection_decision: str
    rewritten_query: str
    retry_count: int
    web_search_used: bool
    reflection_grounded: bool
    reflection_complete: bool
    reflection_relevant: bool
    reflection_fresh: bool
    reflection_issue_source: str
    reflection_rationale: str
    retrieval_assessment_summary: str
    retrieval_used_doc_indexes: list[int]
    retrieval_discarded_doc_indexes: list[int]
    retrieval_document_assessments: list[dict[str, Any]]


class ReflectionResult(BaseModel):
    decision: Literal["answer", "retrieve_more"] = Field(
        description="Return answer if grounded enough, otherwise retrieve_more."
    )
    grounded: bool = Field(description="Whether the draft answer is supported by the retrieved context.")
    complete: bool = Field(description="Whether the draft answer sufficiently covers the user's request.")
    relevant: bool = Field(description="Whether the draft answer is relevant to the user's request.")
    fresh: bool = Field(description="Whether the current evidence appears fresh enough for the user's request.")
    issue_source: Literal[
        "none",
        "query_problem",
        "retrieval_problem",
        "freshness_problem",
        "answer_problem",
        "mixed",
    ] = Field(description="Primary reason the draft is weak, if any.")
    rationale: str = Field(description="Why the current answer is sufficiently supported or not.")
    rewritten_query: str = Field(
        description="Better retrieval query for the next pass. Empty if not needed."
    )


class RetrievedDocumentAssessment(BaseModel):
    doc_index: int = Field(description="1-based document index in the retrieved context.")
    source: str = Field(description="Document source identifier.")
    relevance_score: int = Field(ge=0, le=5, description="Coarse relevance score from 0 to 5.")
    use: bool = Field(description="Whether this document should be kept for answer generation.")
    rationale: str = Field(description="Short reason in Korean.")


class RetrievalAssessmentResult(BaseModel):
    summary: str = Field(description="Overall retrieval quality summary in Korean.")
    documents: list[RetrievedDocumentAssessment] = Field(default_factory=list)


class ReflectionAssessment(BaseModel):
    grounded: bool
    complete: bool
    relevant: bool
    fresh: bool
    issue_source: str
    rationale: str


class RetrievalAssessment(BaseModel):
    summary: str
    used_doc_indexes: list[int] = Field(default_factory=list)
    discarded_doc_indexes: list[int] = Field(default_factory=list)
    documents: list[RetrievedDocumentAssessment] = Field(default_factory=list)


class SelfRAGRequest(BaseModel):
    question: str = Field(min_length=1, description="User question to answer.")
    chat_history: list[ChatTurn] = Field(
        default_factory=list,
        description="Prior conversation turns supplied by the caller.",
    )
    include_trace: bool = Field(
        default=True,
        description="Whether to include structured execution trace in the response.",
    )


class RetrievedDocument(BaseModel):
    source: str
    content: str


class TraceEvent(BaseModel):
    stage: str
    message: str
    elapsed_ms: int
    details: dict[str, Any] = Field(default_factory=dict)


class SelfRAGResponse(BaseModel):
    answer: str
    current_query: str
    retry_count: int
    reflection_decision: str
    reflection: ReflectionAssessment | None = None
    retrieval: RetrievalAssessment | None = None
    documents: list[RetrievedDocument]
    trace: list[TraceEvent] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
