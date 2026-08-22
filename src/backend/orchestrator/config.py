"""
Central configuration for the Evaluator-Generator platform.
Loads everything from environment variables (.env) so no secrets are hardcoded.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    model_name: str = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-20b:free")
    base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


@dataclass(frozen=True)
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = _int("REDIS_PORT", 6379)
    db: int = _int("REDIS_DB", 0)
    password: str = os.getenv("REDIS_PASSWORD", "") or None
    ttl_embeddings: int = _int("REDIS_TTL_EMBEDDINGS", 604800)
    ttl_llm_response: int = _int("REDIS_TTL_LLM_RESPONSE", 86400)
    ttl_evaluation: int = _int("REDIS_TTL_EVALUATION", 86400)
    ttl_retrieval: int = _int("REDIS_TTL_RETRIEVAL", 3600)


@dataclass(frozen=True)
class LoopConfig:
    max_iterations: int = _int("MAX_LOOP_ITERATIONS", 4)


@dataclass(frozen=True)
class LogConfig:
    level: str = os.getenv("LOG_LEVEL", "INFO")
    file: str = os.getenv("LOG_FILE", "logs/app.log")


LLM = LLMConfig()
REDIS = RedisConfig()
LOOP = LoopConfig()
LOGGING = LogConfig()
