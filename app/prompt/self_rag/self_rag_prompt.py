GENERATE_ANSWER_PROMPT = """
You are answering the user's question using retrieved evidence.

Rules:
- Use the retrieved context first.
- Consider the recent chat history for follow-up questions.
- If evidence is incomplete, say what is uncertain.
- Do not invent unsupported facts.
- Write the answer in Korean.

Recent chat history:
{chat_history}

Question:
{question}

Context:
{context}
"""

REFLECT_ON_ANSWER_PROMPT = """
You are a self-reflection module for a retrieval-augmented assistant.

Review:
- recent chat history
- the user question
- current retrieval query
- retrieved context
- draft answer

Return:
- decision="answer" if the draft is sufficiently supported.
- decision="retrieve_more" if the draft is weak, unsupported, incomplete,
  or likely needs broader/fresher evidence.

If decision is retrieve_more, provide a rewritten_query.

Recent chat history:
{chat_history}

Question:
{question}

Current query:
{current_query}

Retrieved context:
{context}

Draft answer:
{generation}
"""

REVISE_ANSWER_PROMPT = """
You are rewriting the final answer after retrieval attempts are exhausted.

Goal:
- Produce the safest and most useful final answer in Korean.
- Use only information supported by the retrieved context.
- Remove speculative or weakly supported claims from the previous draft.
- Keep helpful details that are grounded in the context.
- Explicitly mention uncertainty or missing evidence when needed.
- Do not ask for another retrieval step.

Recent chat history:
{chat_history}

Question:
{question}

Retrieved context:
{context}

Previous draft answer:
{generation}

Write the revised final answer only.
"""
