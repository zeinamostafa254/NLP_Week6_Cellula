"""
Evaluator-Generator Orchestrator (LCEL-based)
=============================================
Connects the pipeline: Retrieve -> Generate -> Evaluate in a bounded
feedback loop using LangChain LCEL (RunnableLambda + pipe operator).

Uses Member 2's run_generator() and run_evaluator() directly — no
protocols, no interfaces, no adapters needed.

Redis caching is transparently consulted at each step.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List

from langchain_core.runnables import RunnableLambda

from .cache import cache
from .config import LOOP

# Member 1's retrieval function
from src.backend.core.ingestion import retrieve_context

# Member 2's generator and evaluator functions
from src.backend.llm.llm_core import run_generator, run_evaluator

logger = logging.getLogger("evalgen.orchestrator")


@dataclass
class LoopResult:
    """Result of the feedback loop, returned to the API/UI."""
    final_answer: str
    accepted: bool
    iterations_used: int
    history: List[dict] = field(default_factory=list)
    truncated: bool = False


class EvaluatorGeneratorOrchestrator:
    def __init__(self, max_iterations: int = LOOP.max_iterations):
        self.max_iterations = max_iterations

        # Wrap each step as a LangChain Runnable
        self._retrieve_step = RunnableLambda(self._retrieve)
        self._generate_step = RunnableLambda(self._generate)
        self._evaluate_step = RunnableLambda(self._evaluate)

        # LCEL composition: one full pass = retrieve -> generate -> evaluate
        self.single_pass_chain = (
            self._retrieve_step | self._generate_step | self._evaluate_step
        )

    # ----------------  LCEL steps ----------------

    def _retrieve(self, state: dict) -> dict:
        """Retrieve relevant context from ChromaDB (with Redis cache)."""
        question = state["question"]

        cached = cache.get_retrieval(question)
        if cached is not None:
            logger.info("Retrieval cache HIT")
            state["context"] = cached
            return state

        # Call Member 1's retrieve_context — returns a str
        context = retrieve_context(question)
        cache.set_retrieval(question, context)
        state["context"] = context
        return state

    def _generate(self, state: dict) -> dict:
        """Generate an answer using Member 2's run_generator (with Redis cache)."""
        question = state["question"]
        context = state["context"]
        feedback = state.get("feedback", "None")

        cached = cache.get_llm_response(question, context, feedback)
        if cached is not None:
            logger.info("Generator cache HIT (iteration %s)", state["iteration"])
            state["answer"] = cached
            return state

        # Call Member 2's run_generator — takes (question: str, context: str, feedback: str)
        answer = run_generator(question=question, context=context, feedback=feedback)
        cache.set_llm_response(question, context, answer, feedback)

        state["answer"] = answer
        return state

    def _evaluate(self, state: dict) -> dict:
        """Evaluate the answer using Member 2's run_evaluator (with Redis cache)."""
        question = state["question"]
        context = state["context"]
        answer = state["answer"]

        cached = cache.get_evaluation(question, answer)
        if cached is not None:
            logger.info("Evaluation cache HIT (iteration %s)", state["iteration"])
            state["is_satisfactory"] = cached["is_satisfactory"]
            state["feedback"] = cached["feedback"]
            return state

        # Call Member 2's run_evaluator — returns EvaluationResult(is_satisfactory, feedback)
        result = run_evaluator(question=question, context=context, generated_answer=answer)

        cache.set_evaluation(question, answer, {
            "is_satisfactory": result.is_satisfactory,
            "feedback": result.feedback,
        })

        state["is_satisfactory"] = result.is_satisfactory
        state["feedback"] = result.feedback
        return state

    # ---------------- the bounded feedback loop ----------------

    def run(self, question: str) -> LoopResult:
        """
        Execute: Retrieve -> Generate -> Evaluate -> (loop if not satisfactory).
        Capped at max_iterations. Returns the final answer either way.
        """
        history: List[dict] = []
        state = {"question": question, "context": "", "feedback": "None", "iteration": 0}
        latest_answer = ""

        for iteration in range(1, self.max_iterations + 1):
            state["iteration"] = iteration
            logger.info("Loop iteration %s/%s for question: %r", iteration, self.max_iterations, question)

            state = self.single_pass_chain.invoke(state)

            latest_answer = state["answer"]
            is_satisfactory = state["is_satisfactory"]
            feedback = state["feedback"]

            history.append({
                "iteration": iteration,
                "answer": latest_answer,
                "accepted": is_satisfactory,
                "feedback": feedback,
            })

            if is_satisfactory:
                logger.info("Answer accepted on iteration %s", iteration)
                return LoopResult(
                    final_answer=latest_answer,
                    accepted=True,
                    iterations_used=iteration,
                    history=history,
                    truncated=False,
                )

            # Feed the evaluator's critique back to the generator
            state["feedback"] = feedback

        # Max iterations reached without acceptance
        logger.warning("Max iterations (%s) reached for question: %r", self.max_iterations, question)
        return LoopResult(
            final_answer=latest_answer,
            accepted=False,
            iterations_used=self.max_iterations,
            history=history,
            truncated=True,
        )
