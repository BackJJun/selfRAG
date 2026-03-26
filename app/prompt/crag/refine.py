CRAG_REFINE_EVIDENCE_PROMPT = """
You are refining retrieved documents into a compact evidence set for a corrective RAG workflow.

Review:
- recent chat history
- user question
- current retrieval query
- retrieval quality summary
- filtered retrieved documents

Return a structured result with:
- summary: concise Korean explanation of the refined evidence set
- quality: high, medium, or low
- items: evidence items with
  - doc_index
  - source
  - claim
  - support_text
  - relevance_score (0-5)
  - confidence (high, medium, low)

Rules:
- Extract only question-relevant evidence.
- Ignore noisy surrounding text, generic background, and weakly related claims.
- Prefer concrete facts, policy points, product capabilities, and numerical details when relevant.
- Keep support_text short and specific.
- If evidence is thin, return fewer items rather than inventing coverage.
- Write summary, claim, and support_text in Korean.
- Document indexes are 1-based and must match the provided document numbers.

Recent chat history:
{chat_history}

Question:
{question}

Current query:
{current_query}

Retrieval summary:
{retrieval_summary}

Retrieved context:
{context}
"""
