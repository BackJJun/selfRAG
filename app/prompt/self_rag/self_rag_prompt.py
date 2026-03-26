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

Return a structured evaluation with these fields:
- decision: "answer" or "retrieve_more"
- grounded: true if the draft is supported by the retrieved context
- complete: true if the draft sufficiently covers the user's request
- relevant: true if the draft is on-topic and answers the user's request
- fresh: true if the current evidence seems fresh enough for the request
- issue_source: one of
  - none
  - query_problem
  - retrieval_problem
  - freshness_problem
  - answer_problem
  - mixed
- rationale: concise explanation in Korean
- rewritten_query: improved retrieval query when decision is "retrieve_more"

Guidance for issue_source:
- query_problem: the search query is too vague, too broad, or badly focused
- retrieval_problem: the query is acceptable but the retrieved context is weak, sparse, or off-target
- freshness_problem: the user needs fresher evidence than the current context provides
- answer_problem: the context is usable but the draft answer itself is poorly synthesized or overclaims
- mixed: multiple problems are present
- none: no major issue

Rules:
- If the user asks about recent or changing information, mark fresh carefully.
- If the current context is not enough to answer confidently, do not pretend it is enough.
- If decision is "retrieve_more", rewritten_query should be specific and actionable.
- Write rationale in Korean.

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
