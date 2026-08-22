"""
Default memory implementations, isolated per session and per agent.

These satisfy the GeneratorMemory / EvaluatorMemory protocols in interfaces.py
so the orchestrator can be built and tested end-to-end right now.

IMPORTANT: Member 2 already has GeneratorMemory / EvaluatorMemory classes.
When you get their real code, swap it in at the two call sites in
orchestrator.py (`GeneratorMemory(session_id)` / `EvaluatorMemory(session_id)`)
-- as long as their classes expose the same methods (add_turn/add_evaluation/
get_history/clear), nothing else in this file changes.

Isolation guarantee: each memory is stored under a namespaced Redis key
(f"mem:generator:{session_id}" vs f"mem:evaluator:{session_id}"), and each
class only ever reads/writes its own namespace. There is no shared object
reference between the two, so accidental cross-access is structurally
impossible, not just a convention.
"""
from __future__ import annotations
import json
import logging
from typing import List, Optional

from .config import REDIS
from .cache import cache

logger = logging.getLogger("evalgen.memory")

_MEMORY_TTL = 60 * 60 * 6  # 6 hours of conversation retention per session


class _BaseMemory:
    """Not exposed directly -- Generator/EvaluatorMemory below set their own prefix."""

    _prefix: str = "mem:base"

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._key = f"{self._prefix}:{session_id}"

    def _load(self) -> list:
        if not cache.available:
            return getattr(self, "_local_fallback", [])
        raw = cache._get(self._key)  # noqa: SLF001 - internal reuse within same package
        return json.loads(raw) if raw else []

    def _save(self, history: list) -> None:
        if not cache.available:
            self._local_fallback = history
            return
        cache._set(self._key, json.dumps(history), _MEMORY_TTL)  # noqa: SLF001

    def get_history(self) -> list:
        return self._load()

    def clear(self) -> None:
        self._save([])


class GeneratorMemory(_BaseMemory):
    _prefix = "mem:generator"

    def add_turn(self, question: str, context: List[str], answer: str) -> None:
        history = self._load()
        history.append({
            "type": "turn",
            "question": question,
            "context": context,
            "answer": answer,
        })
        self._save(history)

    def add_improvement_attempt(self, feedback: str, revised_answer: str) -> None:
        history = self._load()
        history.append({
            "type": "improvement",
            "feedback": feedback,
            "revised_answer": revised_answer,
        })
        self._save(history)


class EvaluatorMemory(_BaseMemory):
    _prefix = "mem:evaluator"

    def add_evaluation(self, question: str, answer: str, verdict: str, feedback: str) -> None:
        history = self._load()
        history.append({
            "type": "evaluation",
            "question": question,
            "answer": answer,
            "verdict": verdict,
            "feedback": feedback,
        })
        self._save(history)
