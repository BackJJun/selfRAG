CRAG_ANSWER_QUALITY_PROMPT = """
You are evaluating a draft answer in a corrective RAG workflow.

Review:
- recent chat history
- user question
- current retrieval query
- refined evidence
- draft answer

Return a structured result with:
- grounded
- complete
- relevant
- rationale
- missing_points: short Korean list
- unsupported_claims: short Korean list
- next_action: one of finalize, regenerate, rewrite_query, web_search, revise
- rewritten_query: optional improved query when rewrite_query is chosen

Action guidance:
- finalize: the answer is grounded, relevant, and sufficiently complete
- regenerate: the evidence is usable but the draft synthesis is weak or poorly organized
- rewrite_query: the retrieval focus is wrong or key evidence is missing
- web_search: fresher or broader external evidence is required
- revise: the draft is mostly usable but needs conservative cleanup to remove overclaims

Rules:
- Penalize claims without explicit support in the documents.
- Prefer regenerate over revise when the answer structure is weak but evidence is usable.
- Prefer revise when the answer is mostly correct but contains a few risky claims.
- Use Korean for rationale, missing_points, and unsupported_claims.

Recent chat history:
{chat_history}

Question:
{question}

Current query:
{current_query}

Refined evidence:
{context}

Draft answer:
{generation}
"""


CRAG_FINAL_ANSWER_PROMPT = """
You are the final approval step in a corrective RAG workflow.

Review:
- user question
- refined evidence
- current answer
- latest answer assessment

Return a structured result with:
- approved: true or false
- action: end or revise
- rationale: concise Korean explanation

Rules:
- Approve only when the answer is grounded, relevant, and sufficiently complete.
- If the answer is still risky, incomplete, or overclaiming, choose revise.
- Do not request another retrieval step here.
- Write rationale in Korean.

Question:
{question}

Refined evidence:
{context}

Current answer:
{generation}

Latest answer assessment:
{answer_assessment}
"""
