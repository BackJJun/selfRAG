from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ChatTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class BaseGraphState(TypedDict):
    question: str
    current_query: str
    chat_history: list[ChatTurn]
    documents: list
    generation: str
    rewritten_query: str
    retry_count: int
    web_search_used: bool
    retrieval_used_doc_indexes: list[int]
    retrieval_discarded_doc_indexes: list[int]
    retrieval_document_assessments: list[dict[str, Any]]
    refined_evidence: list[dict[str, Any]]
    refine_summary: str
    refine_quality: str
    evidence_count: int


class GraphState(BaseGraphState):
    reflection_decision: str
    reflection_grounded: bool
    reflection_complete: bool
    reflection_relevant: bool
    reflection_fresh: bool
    reflection_issue_source: str
    reflection_rationale: str
    retrieval_assessment_summary: str


class CRAGGraphState(BaseGraphState):
    retrieval_quality: str
    retrieval_score: int
    retrieval_issue_type: str
    retrieval_reason: str
    retrieval_should_retry: bool
    retrieval_should_use_web: bool
    answer_grounded: bool
    answer_complete: bool
    answer_relevant: bool
    answer_rationale: str
    answer_next_action: str
    answer_missing_points: list[str]
    answer_unsupported_claims: list[str]
    final_answer_approved: bool
    final_answer_action: str
    final_answer_rationale: str
    correction_retry_count: int
    final_revision_count: int


class HybridGraphState(BaseGraphState):
    reflection_decision: str
    reflection_grounded: bool
    reflection_complete: bool
    reflection_relevant: bool
    reflection_fresh: bool
    reflection_issue_source: str
    reflection_rationale: str
    retrieval_quality: str
    retrieval_score: int
    retrieval_issue_type: str
    retrieval_reason: str
    retrieval_should_retry: bool
    retrieval_should_use_web: bool


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


class CRAGRetrievalAssessmentResult(BaseModel):
    quality: Literal["high", "medium", "low"] = Field(description="Overall retrieval quality tier.")
    score: int = Field(ge=0, le=100, description="Overall retrieval quality score.")
    issue_type: Literal[
        "none",
        "query_problem",
        "retrieval_noise",
        "retrieval_gap",
        "freshness_needed",
        "mixed",
    ] = Field(description="Primary retrieval issue type.")
    should_retry_retrieval: bool = Field(description="Whether another local retrieval pass is useful.")
    should_use_web: bool = Field(description="Whether external web evidence is needed.")
    rewritten_query: str = Field(description="A more focused retrieval query when retry is needed.")
    summary: str = Field(description="Overall retrieval diagnosis in Korean.")
    documents: list[RetrievedDocumentAssessment] = Field(default_factory=list)


class CRAGAnswerAssessmentResult(BaseModel):
    grounded: bool = Field(description="Whether the draft answer is grounded in the current evidence.")
    complete: bool = Field(description="Whether the draft answer covers the user's request sufficiently.")
    relevant: bool = Field(description="Whether the draft answer stays on topic.")
    rationale: str = Field(description="Answer quality diagnosis in Korean.")
    missing_points: list[str] = Field(default_factory=list, description="Missing aspects in Korean.")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Unsupported or weakly supported claims in Korean.",
    )
    next_action: Literal[
        "finalize",
        "regenerate",
        "rewrite_query",
        "web_search",
        "revise",
    ] = Field(description="Recommended corrective action.")
    rewritten_query: str = Field(description="Optional improved query when rewrite_query is selected.")


class CRAGEvidenceItem(BaseModel):
    doc_index: int = Field(description="1-based document index where the evidence came from.")
    source: str = Field(description="Document source identifier.")
    claim: str = Field(description="Question-relevant evidence claim in Korean.")
    support_text: str = Field(description="Short supporting text or excerpt in Korean.")
    relevance_score: int = Field(ge=0, le=5, description="Evidence relevance score from 0 to 5.")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence in this evidence item.")


class CRAGRefineEvidenceResult(BaseModel):
    summary: str = Field(description="Evidence refinement summary in Korean.")
    quality: Literal["high", "medium", "low"] = Field(description="Quality of the refined evidence set.")
    items: list[CRAGEvidenceItem] = Field(default_factory=list, description="Refined evidence items.")


class CRAGFinalAssessmentResult(BaseModel):
    approved: bool = Field(description="Whether the current answer is safe to return.")
    action: Literal["end", "revise"] = Field(description="Final action after the last evaluation.")
    rationale: str = Field(description="Final answer approval rationale in Korean.")


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


class CRAGAnswerAssessment(BaseModel):
    grounded: bool
    complete: bool
    relevant: bool
    rationale: str
    next_action: str
    missing_points: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


class CRAGFinalAssessment(BaseModel):
    approved: bool
    action: str
    rationale: str


class CRAGRetrievalAssessment(BaseModel):
    quality: str
    score: int
    issue_type: str
    summary: str
    used_doc_indexes: list[int] = Field(default_factory=list)
    discarded_doc_indexes: list[int] = Field(default_factory=list)
    documents: list[RetrievedDocumentAssessment] = Field(default_factory=list)


class CRAGRefineAssessment(BaseModel):
    summary: str
    quality: str
    evidence_count: int
    items: list[CRAGEvidenceItem] = Field(default_factory=list)


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
    pipeline: Literal["self_rag", "crag", "hybrid_rag"] = Field(
        default="self_rag",
        description="Which pipeline to run.",
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


class CRAGResponse(BaseModel):
    answer: str
    current_query: str
    retry_count: int
    correction_retry_count: int
    retrieval: CRAGRetrievalAssessment | None = None
    refine: CRAGRefineAssessment | None = None
    answer_assessment: CRAGAnswerAssessment | None = None
    final_assessment: CRAGFinalAssessment | None = None
    documents: list[RetrievedDocument]
    trace: list[TraceEvent] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class HybridRAGResponse(BaseModel):
    answer: str
    current_query: str
    retry_count: int
    retrieval: CRAGRetrievalAssessment | None = None
    refine: CRAGRefineAssessment | None = None
    reflection: ReflectionAssessment | None = None
    documents: list[RetrievedDocument]
    trace: list[TraceEvent] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
