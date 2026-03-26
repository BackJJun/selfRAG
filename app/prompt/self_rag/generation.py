GENERATE_ANSWER_PROMPT = """
You are answering the user's question using retrieved evidence.

Rules:
- Use only the retrieved context.
- Consider the recent chat history for follow-up questions.
- If evidence is incomplete, say what is uncertain.
- Do not invent unsupported facts.
- Write the answer in Korean.
- For each paragraph or major bullet, add evidence tags like [Document 1].
- Use only document numbers that actually support the statement.
- If multiple documents support the same statement, cite multiple tags.

Recent chat history:
{chat_history}

Question:
{question}

Context:
{context}
"""
