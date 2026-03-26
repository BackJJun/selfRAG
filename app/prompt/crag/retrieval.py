CRAG_RETRIEVAL_QUALITY_PROMPT = """
You are evaluating retrieval quality for a corrective RAG workflow.

Review:
- recent chat history
- user question
- current retrieval query
- retrieved documents

Return a structured result with:
- quality: high, medium, or low
- score: integer 0 to 100
- issue_type: one of
  - none
  - query_problem
  - retrieval_noise
  - retrieval_gap
  - freshness_needed
  - mixed
- should_retry_retrieval: whether local retrieval should be retried
- should_use_web: whether external web evidence is needed
- rewritten_query: improved query when retrying retrieval helps
- summary: concise Korean explanation
- documents: per-document judgments with
  - doc_index
  - source
  - relevance_score (0-5)
  - use
  - rationale in Korean

Rules:
- Be strict about noisy or weakly related documents.
- Keep documents that provide direct evidence for key parts of the question.
- If the question asks for recent or changing facts, treat freshness as important.
- Prefer a focused rewrite over vague retries.
- Document indexes are 1-based and must match the provided numbering.
- Write summary and document rationales in Korean.

Recent chat history:
{chat_history}

Question:
{question}

Current query:
{current_query}

Retrieved documents:
{context}
"""
