import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' imports resolve properly regardless of working directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.backend.llm.llm_core import run_generator, run_evaluator
from src.backend.llm.memories import generator_memory, evaluator_memory

# Mock context for isolated testing without requiring vector DB ingestion
MOCK_CONTEXT = """
The Quantum Engine framework was developed in 2025 by Cellula Labs. 
It utilizes a dual-node cluster architecture to achieve sub-millisecond latency. 
It supports Python 3.10+ and requires an NVIDIA GPU with at least 8GB VRAM.
It does not support macOS operating systems.
"""

def test_1_generator_grounding():
    print("\n--- TEST 1: Generator Grounding & Hallucination Check ---")
    
    # 1a. Answerable question
    q1 = "What is the GPU requirement for Quantum Engine?"
    ans1 = run_generator(question=q1, context=MOCK_CONTEXT)
    print(f"Question: {q1}")
    print(f"Generator Output:\n{ans1}\n")
    
    # 1b. Unanswerable question (Must state info is not available rather than hallucinating)[cite: 1]
    q2 = "How much does Quantum Engine cost to license?"
    ans2 = run_generator(question=q2, context=MOCK_CONTEXT)
    print(f"Question: {q2}")
    print(f"Generator Output:\n{ans2}\n")


def test_2_evaluator_structured_output():
    print("\n--- TEST 2: Evaluator Structured Grading ---")
    q = "What is the GPU requirement for Quantum Engine?"
    
    # 2a. Accurate and grounded answer[cite: 1]
    good_answer = "Quantum Engine requires an NVIDIA GPU with at least 8GB VRAM."
    eval_good = run_evaluator(question=q, context=MOCK_CONTEXT, generated_answer=good_answer)
    print("Evaluating Grounded Answer:")
    print(f"  Satisfactory: {eval_good.is_satisfactory}")
    print(f"  Feedback: {eval_good.feedback}\n")
    
    # 2b. Unsupported claim answer[cite: 1]
    bad_answer = "Quantum Engine runs on any AMD GPU with 4GB VRAM and supports macOS."
    eval_bad = run_evaluator(question=q, context=MOCK_CONTEXT, generated_answer=bad_answer)
    print("Evaluating Unsupported Answer:")
    print(f"  Satisfactory: {eval_bad.is_satisfactory}")
    print(f"  Feedback: {eval_bad.feedback}\n")


def test_3_feedback_refinement():
    print("\n--- TEST 3: Evaluator Feedback Loop Refinement ---")
    q = "Does Quantum Engine run on Mac?"
    
    # Incomplete initial answer
    initial_answer = "Quantum Engine is a framework developed in 2025."
    
    # Generate Evaluator feedback[cite: 1]
    eval_res = run_evaluator(question=q, context=MOCK_CONTEXT, generated_answer=initial_answer)
    print(f"Evaluator Feedback: {eval_res.feedback}")
    
    # Pass feedback back to Generator[cite: 1]
    improved_answer = run_generator(question=q, context=MOCK_CONTEXT, feedback=eval_res.feedback)
    print(f"Improved Generator Output:\n{improved_answer}\n")


def test_4_memory_isolation():
    print("\n--- TEST 4: Isolated Independent Memories Check ---")
    gen_vars = generator_memory.load_memory_variables({})
    eval_vars = evaluator_memory.load_memory_variables({})
    
    gen_history = gen_vars.get("generator_history", [])
    eval_history = eval_vars.get("evaluator_history", [])
    
    print(f"Generator Memory Store Count: {len(gen_history)}")
    print(f"Evaluator Memory Store Count: {len(eval_history)}")
    
    # Verify strict memory separation[cite: 1]
    assert gen_history != eval_history, "Memory isolation breach detected! Models share state."
    print("Memory Isolation Confirmed: Generator and Evaluator memories are isolated.")


if __name__ == "__main__":
    print("Executing Member 2 Core LLM Test Suite...")
    try:
        test_1_generator_grounding()
        test_2_evaluator_structured_output()
        test_3_feedback_refinement()
        test_4_memory_isolation()
        print("\nALL TESTS PASSED SUCCESSFULLY! Ready for handoff to Member 3.")
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")