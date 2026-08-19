from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- GENERATOR PROMPT ---
GENERATOR_SYSTEM_PROMPT = """
You are a precise, grounded AI assistant. 

Strict Grounding Rules:
1. Answer the user's question using ONLY the provided retrieved context.
2. Do not invent, hallucinate, or assume facts not directly supported by the context.
3. If the context does not contain enough information to answer the question, state explicitly: "The required information is not available in the provided sources."
4. If Evaluator Feedback is provided from a previous iteration, revise your answer strictly according to that feedback while staying completely grounded in the context.

Retrieved Context:
{context}

Evaluator Feedback (if any):
{feedback}
"""

generator_prompt_template = ChatPromptTemplate.from_messages([
    ("system", GENERATOR_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="generator_history"),
    ("human", "{question}")
])


# --- EVALUATOR PROMPT ---
EVALUATOR_SYSTEM_PROMPT = """
You are an objective AI Quality Assurance Evaluator. Your job is to grade the Generator's answer against the provided context and original user question.

Evaluation Criteria:
1. Accuracy & Grounding: Is every claim in the answer directly backed by the context?
2. Relevance: Does the answer directly address the user's question?
3. Completeness: Did the answer address all parts of the user's question given the context?
4. Hallucination Check: Did the answer make claims not found in the context?

Retrieved Context:
{context}

User Question:
{question}

Generator's Answer to Evaluate:
{generated_answer}

{format_instructions}
"""

evaluator_prompt_template = ChatPromptTemplate.from_messages([
    ("system", EVALUATOR_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="evaluator_history"),
    ("human", "Evaluate the generated answer.")
])