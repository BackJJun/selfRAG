REWRITE_QUERY_PROMPT = """
Rewrite the user's question into a better retrieval query.

Requirements:
- Keep it concise.
- Preserve the original intent.
- Reflect the recent chat context if it matters.
- Add useful search keywords.
- Return only the rewritten query.

Recent chat history:
{chat_history}

Original question:
{question}

Current query:
{current_query}
"""
