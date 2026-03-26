EVALUATE_RETRIEVED_DOCUMENTS_PROMPT = """
You are evaluating retrieved documents for a retrieval-augmented assistant.

Review:
- the recent chat history
- the user question
- the current retrieval query
- the retrieved documents

For each retrieved document, decide:
- relevance_score: integer from 0 to 5
- use: true if this document should be kept for answer generation
- rationale: short Korean explanation

Rules:
- Be strict about off-topic or weakly related documents.
- Keep documents that are useful evidence even if they cover only part of the question.
- Prefer precision over recall.
- Return a short overall summary in Korean.
- Document indexes are 1-based and must match the provided document numbers.

Recent chat history:
{chat_history}

Question:
{question}

Current query:
{current_query}

Retrieved documents:
{context}
"""
