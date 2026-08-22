"""
Demo entrypoint for Member 3's part. Run:

    python -m src.backend.orchestrator.main "What is the Redis caching layer used for?"

This wires together:
  - a retriever  (stub for now -- swap for Member 1's real vector store retriever)
  - generator/evaluator chains (real OpenRouter calls, placeholder prompts --
    swap prompts for Member 2's real ones in stubs.py or replace the classes
    entirely with Member 2's real GeneratorChain/EvaluatorChain)
  - isolated memories (memory.py)
  - Redis cache (cache.py)
  - the bounded feedback-loop orchestrator (orchestrator.py)

Once Member 1 and Member 2 hand over real code, only THIS file (and possibly
stubs.py) should need to change -- orchestrator.py, cache.py, memory.py,
config.py stay untouched, since they're built against interfaces.py.
"""
import sys
import uuid

from .logging_config import setup_logging
from .memory import GeneratorMemory, EvaluatorMemory
from .orchestrator import EvaluatorGeneratorOrchestrator
from .stubs import FakeRetriever, RealGeneratorChain, RealEvaluatorChain

setup_logging()


def build_orchestrator(session_id: str) -> EvaluatorGeneratorOrchestrator:
    return EvaluatorGeneratorOrchestrator(
        retriever=FakeRetriever(),                 # <- replace with Member 1's retriever
        generator=RealGeneratorChain(),             # <- replace with Member 2's GeneratorChain
        evaluator=RealEvaluatorChain(),             # <- replace with Member 2's EvaluatorChain
        generator_memory=GeneratorMemory(session_id),
        evaluator_memory=EvaluatorMemory(session_id),
    )


def main():
    question = " ".join(sys.argv[1:]) or "What does the Redis caching layer store?"
    session_id = str(uuid.uuid4())

    orchestrator = build_orchestrator(session_id)
    result = orchestrator.run(question)

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)
    for step in result.history:
        status = "ACCEPTED" if step["accepted"] else "REVISE"
        print(f"\n--- Iteration {step['iteration']} [{status}] ---")
        print(f"Answer: {step['answer']}")
        if step["feedback"]:
            print(f"Feedback: {step['feedback']}")

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(result.final_answer)
    print(f"\n(accepted={result.accepted}, iterations_used={result.iterations_used}, truncated={result.truncated})")


if __name__ == "__main__":
    main()
