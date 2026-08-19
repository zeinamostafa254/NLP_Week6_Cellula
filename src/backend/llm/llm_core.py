import json
import re
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from src.config.settings import llm
from src.backend.llm.memories import generator_memory, evaluator_memory
from src.backend.llm.prompts import generator_prompt_template, evaluator_prompt_template


# Structured JSON output schema for Evaluator
class EvaluationResult(BaseModel):
    is_satisfactory: bool = Field(
        description="True if the answer is grounded, accurate, complete, and relevant. False otherwise."
    )
    feedback: str = Field(
        description="Detailed improvement suggestions if False, or 'Approved' if True."
    )


# Set up Pydantic output parser for evaluator[cite: 1]
evaluator_parser = PydanticOutputParser(pydantic_object=EvaluationResult)

# --- Chains Definition ---
generator_chain = generator_prompt_template | llm | StrOutputParser()
raw_evaluator_chain = evaluator_prompt_template | llm | StrOutputParser()


def parse_evaluator_response(text: str) -> EvaluationResult:
    """Robustly parse LLM evaluation response into EvaluationResult Pydantic object."""
    # Attempt 1: Standard Pydantic parsing
    try:
        return evaluator_parser.parse(text)
    except Exception:
        pass

    # Attempt 2: Extract JSON block via regex if model wrapped in markdown
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return EvaluationResult(**data)
        except Exception:
            pass

    # Attempt 3: Parse markdown key-value text (e.g. **is_satisfactory:** True)
    is_sat = True
    if (
        re.search(r"is_satisfactory\s*:\s*false", text, re.IGNORECASE)
        or re.search(r"is_satisfactory\s*:\s*no", text, re.IGNORECASE)
        or "unsatisfactory" in text.lower()
        or "needs improvement" in text.lower()
    ):
        is_sat = False

    feedback_match = re.search(r"feedback\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if feedback_match:
        feedback = feedback_match.group(1).strip()
    else:
        feedback = "Approved" if is_sat else text.strip()

    return EvaluationResult(is_satisfactory=is_sat, feedback=feedback)


# --- Core Execution Functions ---

def run_generator(question: str, context: str, feedback: str = "None") -> str:
    """Generates a grounded answer using retrieved context and past feedback."""
    gen_vars = generator_memory.load_memory_variables({"question": question})
    history = gen_vars.get("generator_history", [])

    answer = generator_chain.invoke({
        "question": question,
        "context": context,
        "feedback": feedback,
        "generator_history": history
    })

    generator_memory.save_context(
        {"question": f"Question: {question} (Feedback applied: {feedback})"},
        {"output": answer}
    )
    return answer


def run_evaluator(question: str, context: str, generated_answer: str) -> EvaluationResult:
    """Evaluates the generated answer against criteria and returns structured output."""
    eval_vars = evaluator_memory.load_memory_variables({"question": question})
    history = eval_vars.get("evaluator_history", [])

    # Pass pydantic format instructions to prompt
    raw_response = raw_evaluator_chain.invoke({
        "question": question,
        "context": context,
        "generated_answer": generated_answer,
        "evaluator_history": history,
        "format_instructions": evaluator_parser.get_format_instructions()
    })

    # Safely parse output into EvaluationResult
    result = parse_evaluator_response(raw_response)

    evaluator_memory.save_context(
        {"question": f"Evaluate answer for: {question}"},
        {"output": f"Satisfactory: {result.is_satisfactory} | Feedback: {result.feedback}"}
    )
    return result