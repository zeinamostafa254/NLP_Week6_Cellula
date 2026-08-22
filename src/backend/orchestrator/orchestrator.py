"""
Member 3's centerpiece: connects Generator <-> Evaluator via LCEL and runs
the bounded feedback loop described in the workflow diagram:

    User Question -> Generator -> Answer -> Evaluator -> Decision
        Decision == acceptable      -> Exit, return answer
        Decision == needs work      -> Feedback -> Generator (loop, max 4)
        loop_count == 4 and no pass -> return latest answer + disclaimer

Design notes
------------
- Each stage (retrieve / generate / evaluate) is wrapped as a LangChain
  Runnable (RunnableLambda), and composed with LCEL's `|` operator into a
  single `single_pass_chain`. That chain is what actually gets invoked on
  every iteration -- this is the "LCEL-based orchestration" requirement.
- The *looping* itself is plain Python, because LCEL graphs are static/DAGs
  by design and don't have a first-class "loop until condition" primitive.
  Wrapping a dynamic-length loop in a fake LCEL branch would be a workaround,
  not a chain -- so the loop lives in `run()`, and the LCEL chain is the unit
  of work executed on each of its iterations. This is the standard pattern
  for LCEL + iterative agent loops.
- Redis caching is transparently consulted inside the retrieve/generate/
  evaluate steps, so a repeated question (or a repeated intermediate answer)
  short-circuits an expensive LLM/embedding call.
- Generator and Evaluator each get their OWN memory instance, passed in by
  the caller and never crossed.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.runnables import RunnableLambda

from .cache import cache
from .config import LOOP
from .interfaces import (
    EvaluationResult,
    EvaluatorChain,
    EvaluatorMemory,
    GeneratorChain,
    GeneratorMemory,
    Retriever,
)

logger = logging.getLogger("evalgen.orchestrator")


@dataclass
class LoopResult:
    final_answer: str
    accepted: bool
    iterations_used: int
    history: List[dict] = field(default_factory=list)  # per-iteration trace for UI/logging
    truncated: bool = False  # True if we hit MAX_LOOP_ITERATIONS without acceptance


class EvaluatorGeneratorOrchestrator:
    def __init__(
        self,
        retriever: Retriever,
        generator: GeneratorChain,
        evaluator: EvaluatorChain,
        generator_memory: GeneratorMemory,
        evaluator_memory: EvaluatorMemory,
        max_iterations: int = LOOP.max_iterations,
    ):
        self.retriever = retriever
        self.generator = generator
        self.evaluator = evaluator
        self.generator_memory = generator_memory
        self.evaluator_memory = evaluator_memory
        self.max_iterations = max_iterations

        self._retrieve_step = RunnableLambda(self._retrieve)
        self._generate_step = RunnableLambda(self._generate)
        self._evaluate_step = RunnableLambda(self._evaluate)

        # LCEL composition: one full pass = retrieve -> generate -> evaluate
        self.single_pass_chain = (
            self._retrieve_step | self._generate_step | self._evaluate_step
        )

    # ---------------- individual LCEL steps ----------------

    def _retrieve(self, state: dict) -> dict:
        question = state["question"]
        cached = cache.get_retrieval(question)
        if cached is not None:
            logger.info("Retrieval cache HIT for question hash")
            state["context"] = cached
            return state

        try:
            context = self.retriever.get_relevant_documents(question)
        except Exception:
            logger.exception("Retrieval failed for question: %s", question)
            context = []

        cache.set_retrieval(question, context)
        state["context"] = context
        return state

    def _generate(self, state: dict) -> dict:
        question, context, feedback = state["question"], state["context"], state.get("feedback")

        cached = cache.get_llm_response(question, context, feedback or "")
        if cached is not None:
            logger.info("Generator cache HIT (iteration %s)", state["iteration"])
            answer = cached
        else:
            try:
                answer = self.generator.generate(
                    question=question,
                    context=context,
                    feedback=feedback,
                    memory=self.generator_memory,
                )
            except Exception:
                logger.exception("Generator failed on iteration %s", state["iteration"])
                answer = (
                    "I couldn't generate an answer due to an internal error. "
                    "Please try again."
                )
            cache.set_llm_response(question, context, answer, feedback or "")

        if feedback:
            self.generator_memory.add_improvement_attempt(feedback, answer)
        else:
            self.generator_memory.add_turn(question, context, answer)

        state["answer"] = answer
        return state

    def _evaluate(self, state: dict) -> dict:
        question, answer, context = state["question"], state["answer"], state["context"]

        cached = cache.get_evaluation(question, answer)
        if cached is not None:
            logger.info("Evaluation cache HIT (iteration %s)", state["iteration"])
            result = EvaluationResult(
                is_acceptable=cached["is_acceptable"],
                feedback=cached.get("feedback", ""),
                score=cached.get("score"),
            )
        else:
            try:
                result = self.evaluator.evaluate(
                    question=question, answer=answer, context=context, memory=self.evaluator_memory
                )
            except Exception:
                logger.exception("Evaluator failed on iteration %s", state["iteration"])
                # Fail-safe: if the evaluator itself errors, don't loop forever --
                # treat as "needs improvement" only while iterations remain, otherwise
                # the outer loop's max-iteration guard will terminate cleanly.
                result = EvaluationResult(is_acceptable=False, feedback="Evaluator error; retrying.")

            cache.set_evaluation(
                question,
                answer,
                {"is_acceptable": result.is_acceptable, "feedback": result.feedback, "score": result.score},
            )

        self.evaluator_memory.add_evaluation(
            question=question,
            answer=answer,
            verdict="accept" if result.is_acceptable else "revise",
            feedback=result.feedback,
        )

        state["evaluation"] = result
        return state

    # ---------------- the bounded feedback loop ----------------

    def run(self, question: str) -> LoopResult:
        """
        Executes: Generator -> Evaluator -> (exit | feedback -> Generator), 
        capped at self.max_iterations. Returns the final answer either way.
        """
        history: List[dict] = []
        state = {"question": question, "context": [], "feedback": None, "iteration": 0}
        latest_answer = ""

        for iteration in range(1, self.max_iterations + 1):
            state["iteration"] = iteration
            logger.info("Loop iteration %s/%s for question: %r", iteration, self.max_iterations, question)

            state = self.single_pass_chain.invoke(state)
            evaluation: EvaluationResult = state["evaluation"]
            latest_answer = state["answer"]

            history.append({
                "iteration": iteration,
                "answer": latest_answer,
                "accepted": evaluation.is_acceptable,
                "feedback": evaluation.feedback,
                "score": evaluation.score,
            })

            if evaluation.is_acceptable:
                logger.info("Answer accepted on iteration %s", iteration)
                return LoopResult(
                    final_answer=latest_answer,
                    accepted=True,
                    iterations_used=iteration,
                    history=history,
                    truncated=False,
                )

            # Not acceptable yet -- feed the Evaluator's critique back to the Generator
            state["feedback"] = evaluation.feedback

        # Max loop count reached without acceptance
        logger.warning(
            "Max iterations (%s) reached without an accepted answer for question: %r",
            self.max_iterations, question,
        )
        disclaimer = (
            "\n\n[Note: this answer went through the maximum number of review "
            f"iterations ({self.max_iterations}) and could not be fully validated. "
            "Please verify important details independently.]"
        )
        return LoopResult(
            final_answer=latest_answer + disclaimer,
            accepted=False,
            iterations_used=self.max_iterations,
            history=history,
            truncated=True,
        )
