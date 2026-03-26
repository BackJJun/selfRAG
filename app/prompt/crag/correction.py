CRAG_GENERATE_ANSWER_PROMPT = """
You are generating an answer in a corrective RAG workflow.

Rules:
- Use only the provided refined evidence.
- Consider the recent chat history for follow-up questions.
- Do not invent unsupported facts.
- Write the answer in Korean.
- For each paragraph or major bullet, add evidence tags like [Document 1].
- If evidence is incomplete, state the uncertainty clearly.
- Focus on direct, well-supported statements.

Recent chat history:
{chat_history}

Question:
{question}

Retrieval summary:
{retrieval_summary}

Refine summary:
{refine_summary}

Refined evidence:
{context}
"""


CRAG_REGENERATE_ANSWER_PROMPT = """
You are regenerating an answer after an answer-quality review.

Use:
- recent chat history
- user question
- refined evidence
- previous draft answer
- missing points
- unsupported claims

Rules:
- Fix coverage gaps from missing_points.
- Remove or soften unsupported_claims.
- Use only evidence supported by the refined evidence.
- Write the answer in Korean.
- Keep paragraph or bullet-level evidence tags like [Document 1].
- Do not mention your internal process.

Recent chat history:
{chat_history}

Question:
{question}

Refined evidence:
{context}

Previous draft answer:
{generation}

Missing points:
{missing_points}

Unsupported claims:
{unsupported_claims}
"""


CRAG_REVISE_ANSWER_PROMPT = """
You are revising the final answer conservatively in a corrective RAG workflow.

Rules:
- Remove or soften unsupported claims.
- Keep only statements grounded in the refined evidence.
- Preserve useful evidence tags like [Document 1] where support exists.
- Write the safest helpful answer in Korean.
- Explicitly mention uncertainty when evidence is thin.
- Do not request another retrieval step.

Question:
{question}

Refined evidence:
{context}

Current answer:
{generation}

Unsupported claims:
{unsupported_claims}

Missing points:
{missing_points}
"""
