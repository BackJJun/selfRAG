REVISE_ANSWER_PROMPT = """
You are rewriting the final answer after retrieval attempts are exhausted.

Do this internally before writing the final answer:
1. Identify unsupported or weakly supported claims in the previous draft.
2. Remove them or rewrite them into cautious statements.
3. Keep only claims grounded in the retrieved context.
4. Preserve or repair evidence tags like [Document 1] where support exists.

Output rules:
- Produce the safest and most useful final answer in Korean.
- Use only information supported by the retrieved context.
- Keep the answer helpful but conservative.
- Explicitly mention uncertainty or missing evidence when needed.
- Keep paragraph or bullet-level evidence tags like [Document 1].
- Do not expose your internal analysis steps.
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
