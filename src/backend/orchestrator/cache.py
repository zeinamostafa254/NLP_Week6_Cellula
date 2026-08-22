"""
Redis caching layer.

Caches four kinds of things:
  1. embeddings            -> keyed by hash(text)
  2. retrieval results     -> keyed by hash(query)
  3. LLM responses         -> keyed by hash(question + context + feedback)
  4. evaluations           -> keyed by hash(question + answer)

Context is stored as str (matching Member 2's conventions).

If Redis is unreachable, every method fails OPEN (returns None / no-ops)
instead of crashing the app.
"""
from __future__ import annotations
import hashlib
import json
import logging
from typing import Optional

import redis

from .config import REDIS

logger = logging.getLogger("evalgen.cache")


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


class RedisCache:
    def __init__(self):
        self._client: Optional[redis.Redis] = None
        try:
            self._client = redis.Redis(
                host=REDIS.host,
                port=REDIS.port,
                db=REDIS.db,
                password=REDIS.password,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client.ping()
            logger.info("Connected to Redis at %s:%s", REDIS.host, REDIS.port)
        except Exception as exc:
            logger.warning("Redis unavailable (%s). Running without cache.", exc)
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    # ---------- generic helpers ----------
    def _get(self, key: str) -> Optional[str]:
        if not self._client:
            return None
        try:
            return self._client.get(key)
        except Exception as exc:
            logger.warning("Redis GET failed for %s: %s", key, exc)
            return None

    def _set(self, key: str, value: str, ttl: int) -> None:
        if not self._client:
            return
        try:
            self._client.set(key, value, ex=ttl)
        except Exception as exc:
            logger.warning("Redis SET failed for %s: %s", key, exc)

    # ---------- retrieval results (context as str) ----------
    def get_retrieval(self, query: str) -> Optional[str]:
        return self._get(f"retr:{_hash(query)}")

    def set_retrieval(self, query: str, context: str) -> None:
        self._set(f"retr:{_hash(query)}", context, REDIS.ttl_retrieval)

    # ---------- LLM (generator) responses ----------
    def get_llm_response(self, question: str, context: str, feedback: str = "") -> Optional[str]:
        key = f"gen:{_hash(question, context, feedback or '')}"
        return self._get(key)

    def set_llm_response(self, question: str, context: str, answer: str, feedback: str = "") -> None:
        key = f"gen:{_hash(question, context, feedback or '')}"
        self._set(key, answer, REDIS.ttl_llm_response)

    # ---------- evaluations ----------
    def get_evaluation(self, question: str, answer: str) -> Optional[dict]:
        raw = self._get(f"eval:{_hash(question, answer)}")
        return json.loads(raw) if raw else None

    def set_evaluation(self, question: str, answer: str, result: dict) -> None:
        self._set(f"eval:{_hash(question, answer)}", json.dumps(result), REDIS.ttl_evaluation)

    # ---------- invalidation ----------
    def invalidate_prefix(self, prefix: str) -> int:
        """Use when underlying knowledge changes (e.g. a document is re-uploaded)."""
        if not self._client:
            return 0
        try:
            keys = list(self._client.scan_iter(match=f"{prefix}*"))
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as exc:
            logger.warning("Redis invalidate failed for prefix %s: %s", prefix, exc)
            return 0


cache = RedisCache()
