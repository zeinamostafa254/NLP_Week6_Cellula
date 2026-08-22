"""
Throwaway stand-ins for Member 1 (retriever) and Member 2 (generator/evaluator
prompts), so you can run and test the orchestrator/loop/cache RIGHT NOW,
before their real code is merged in.

Delete this file (or just stop importing it) once real integrations land --
swap in Member 1's actual retriever and Member 2's actual chains in main.py.
"""
from __future__ import annotations
import json
import re
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .interfaces import EvaluationResult
from .llm_client import get_llm


class FakeRetriever:
    """Pretend vector store: just returns a couple of canned chunks."""

    def __init__(self, corpus: Optional[List[str]] = None):
        self.corpus = corpus or [
            "The Evaluator-Generator workflow enforces a max of 4 feedback iterations.",
            "Redis is used to cache embeddings, retrieval results, LLM responses, and evaluations.",
            "The Generator and Evaluator maintain fully isolated memories.",
        ]

    def get_relevant_documents(self, query: str) -> List[str]:
        # naive keyword overlap "retrieval" -- good enough for wiring tests
        q_words = set(re.findall(r"\w+", query.lower()))
        scored = sorted(
            self.corpus,
            key=lambda c: len(q_words & set(re.findall(r"\w+", c.lower()))),
            reverse=True,
        )
        return scored[:3]


class RealGeneratorChain:
    """Actually calls the LLM via OpenRouter, using a placeholder grounding prompt.
    Replace the prompt text with Member 2's real Generator prompt when ready --
    the method signature (generate) is all the orchestrator depends on."""

    def __init__(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a grounded question-answering assistant. Answer ONLY using "
             "the provided context. If the context does not contain the answer, "
             "say explicitly that the information is not available in the provided "
             "sources. Do not invent facts.\n\nContext:\n{context}\n\n"
             "{feedback_block}"),
            ("human", "{question}"),
        ])
        self._chain = prompt | get_llm(temperature=0.2) | StrOutputParser()

    def generate(self, question: str, context: List[str], feedback: Optional[str], memory) -> str:
        feedback_block = (
            f"Your previous answer received this feedback -- revise accordingly:\n{feedback}"
            if feedback else ""
        )
        return self._chain.invoke({
            "question": question,
            "context": "\n".join(context) if context else "(no relevant context found)",
            "feedback_block": feedback_block,
        })


class RealEvaluatorChain:
    """Actually calls the LLM to judge the answer. Replace prompt with Member 2's
    real Evaluator prompt/criteria when ready."""

    def __init__(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a strict answer evaluator. Judge the ANSWER against the "
             "QUESTION and CONTEXT for: accuracy, relevance, completeness, "
             "grounding in context, and absence of unsupported claims.\n"
             "Respond ONLY with compact JSON: "
             '{{"is_acceptable": true|false, "feedback": "<what to improve, or empty if acceptable>"}}'),
            ("human", "QUESTION: {question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"),
        ])
        self._chain = prompt | get_llm(temperature=0.0) | StrOutputParser()

    def evaluate(self, question: str, answer: str, context: List[str], memory) -> EvaluationResult:
        raw = self._chain.invoke({
            "question": question,
            "context": "\n".join(context) if context else "(no relevant context found)",
            "answer": answer,
        })
        try:
            # tolerate models that wrap JSON in prose/backticks
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group(0) if match else raw)
            return EvaluationResult(
                is_acceptable=bool(data.get("is_acceptable", False)),
                feedback=data.get("feedback", ""),
            )
        except Exception:
            # if the evaluator model didn't return valid JSON, fail safe:
            # don't silently accept a possibly-bad answer
            return EvaluationResult(is_acceptable=False, feedback="Evaluator response was not parseable JSON.")
