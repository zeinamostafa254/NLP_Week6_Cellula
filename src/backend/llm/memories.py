from langchain_classic.memory import ConversationBufferMemory

# Generator Memory: Stores user questions, retrieved contexts, and past feedback
generator_memory = ConversationBufferMemory(
    memory_key="generator_history",
    input_key="question",
    return_messages=True
)

# Evaluator Memory: Stores evaluation logs, scoring decisions, and past feedback
evaluator_memory = ConversationBufferMemory(
    memory_key="evaluator_history",
    input_key="question",
    return_messages=True
)