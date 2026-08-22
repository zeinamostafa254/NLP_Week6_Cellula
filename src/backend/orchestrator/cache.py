"""
Redis caching layer.

Caches four kinds of things (per the architecture diagram):
  1. embeddings            -> keyed by hash(text)
  2. retrieval results     -> keyed by hash(query)
  3. LLM responses         -> keyed by hash(question + context + feedback)
  4. evaluations           -> keyed by hash(question + answer)

Cache keys are content-addressed (sha256 of the normalized input), so:
  - identical requests hit the cache regardless of when they were asked
  - if the underlying context/answer changes even slightly, the key changes
    too -> no stale hits. This is the "don't return stale info" requirement.

If Redis is unreachable, every method fails OPEN (returns None / no-ops)
instead of crashing the app, and logs a warning. Caching is a performance
optimization, not a correctness requirement — the system must still work
without it.
"""
from __future__ import annotations
import hashlib
import json
import logging
from typing import Any, List, Optional

import redis

from .config import REDIS

logger = logging.getLogger("evalgen.cache")


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")  # unit separator, avoids "ab"+"c" == "a"+"bc" collisions
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
        except Exception as exc:  # noqa: BLE001 - deliberately broad, cache must fail open
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
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis GET failed for %s: %s", key, exc)
            return None

    def _set(self, key: str, value: str, ttl: int) -> None:
        if not self._client:
            return
        try:
            self._client.set(key, value, ex=ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis SET failed for %s: %s", key, exc)

    # ---------- embeddings ----------
    def get_embedding(self, text: str) -> Optional[List[float]]:
        raw = self._get(f"emb:{_hash(text)}")
        return json.loads(raw) if raw else None

    def set_embedding(self, text: str, embedding: List[float]) -> None:
        self._set(f"emb:{_hash(text)}", json.dumps(embedding), REDIS.ttl_embeddings)

    # ---------- retrieval results ----------
    def get_retrieval(self, query: str) -> Optional[List[str]]:
        raw = self._get(f"retr:{_hash(query)}")
        return json.loads(raw) if raw else None

    def set_retrieval(self, query: str, chunks: List[str]) -> None:
        self._set(f"retr:{_hash(query)}", json.dumps(chunks), REDIS.ttl_retrieval)

    # ---------- LLM (generator) responses ----------
    def get_llm_response(self, question: str, context: List[str], feedback: str = "") -> Optional[str]:
        key = f"gen:{_hash(question, json.dumps(context, sort_keys=True), feedback or '')}"
        return self._get(key)

    def set_llm_response(self, question: str, context: List[str], answer: str, feedback: str = "") -> None:
        key = f"gen:{_hash(question, json.dumps(context, sort_keys=True), feedback or '')}"
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
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis invalidate failed for prefix %s: %s", prefix, exc)
            return 0


cache = RedisCache()  # module-level singleton, imported everywhere
