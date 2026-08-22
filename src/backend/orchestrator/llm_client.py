"""
Single place that builds LangChain chat model instances pointed at OpenRouter.
Both the Generator and Evaluator (Member 2) should import `get_llm()` from here
instead of instantiating their own client, so the whole team only changes
model/provider config in ONE place (config.py / .env).
"""
from langchain_openai import ChatOpenAI

from .config import LLM


def get_llm(temperature: float = 0.2, **kwargs) -> ChatOpenAI:
    """
    Returns a LangChain ChatOpenAI client configured for OpenRouter.
    OpenRouter is OpenAI-API-compatible, so we just override base_url + api_key.

    Usage (Member 2):
        from llm_client import get_llm
        llm = get_llm(temperature=0.3)
        chain = prompt | llm | StrOutputParser()
    """
    if not LLM.api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in."
        )

    return ChatOpenAI(
        model=LLM.model_name,
        api_key=LLM.api_key,
        base_url=LLM.base_url,
        temperature=temperature,
        **kwargs,
    )
