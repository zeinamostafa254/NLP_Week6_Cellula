"""
Demo entrypoint for the orchestrator. Run:

    python -m src.backend.orchestrator.main "Your question here"

Runs the full pipeline: Retrieve -> Generate -> Evaluate (up to 4 iterations).
"""
import sys

from .logging_config import setup_logging
from .orchestrator import EvaluatorGeneratorOrchestrator

setup_logging()


def main():
    question = " ".join(sys.argv[1:]) or "What does this project do?"
    orchestrator = EvaluatorGeneratorOrchestrator()
    result = orchestrator.run(question)

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)
    for step in result.history:
        status = "ACCEPTED" if step["accepted"] else "REVISE"
        print(f"\n--- Iteration {step['iteration']} [{status}] ---")
        print(f"Answer: {step['answer'][:200]}...")
        if step["feedback"]:
            print(f"Feedback: {step['feedback'][:200]}...")

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print(result.final_answer)
    print(f"\n(accepted={result.accepted}, iterations={result.iterations_used}, truncated={result.truncated})")


if __name__ == "__main__":
    main()
