"""
Integration contracts between the three modules.

Member 3 (this orchestrator)
"""
from __future__ import annotations
from typing import List, Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    """Satisfied by Member 1's vector store retriever."""

    def get_relevant_documents(self, query: str) -> List[str]:
        """Return the top-k relevant text chunks for the query."""
        ...


@runtime_checkable
class GeneratorMemory(Protocol):
    """Isolated memory for the Generator LLM only. Must never be shared with Evaluator."""

    def add_turn(self, question: str, context: List[str], answer: str) -> None: ...
    def add_improvement_attempt(self, feedback: str, revised_answer: str) -> None: ...
    def get_history(self) -> list: ...
    def clear(self) -> None: ...


@runtime_checkable
class EvaluatorMemory(Protocol):
    """Isolated memory for the Evaluator LLM only. Must never be shared with Generator."""

    def add_evaluation(self, question: str, answer: str, verdict: str, feedback: str) -> None: ...
    def get_history(self) -> list: ...
    def clear(self) -> None: ...


@runtime_checkable
class GeneratorChain(Protocol):
    """Satisfied by Member 2's Generator LLM chain."""

    def generate(
        self,
        question: str,
        context: List[str],
        feedback: str | None,
        memory: GeneratorMemory,
    ) -> str:
        """Produce an answer grounded ONLY in `context`. `feedback` is the
        Evaluator's critique of the previous attempt (None on first pass)."""
        ...


@runtime_checkable
class EvaluatorChain(Protocol):
    """Satisfied by Member 2's Evaluator LLM chain."""

    def evaluate(
        self,
        question: str,
        answer: str,
        context: List[str],
        memory: EvaluatorMemory,
    ) -> "EvaluationResult":
        ...


class EvaluationResult:
    """Structured verdict returned by the Evaluator."""

    def __init__(self, is_acceptable: bool, feedback: str = "", score: float | None = None):
        self.is_acceptable = is_acceptable
        self.feedback = feedback
        self.score = score

    def __repr__(self) -> str:
        return f"EvaluationResult(is_acceptable={self.is_acceptable}, score={self.score}, feedback={self.feedback!r})"
