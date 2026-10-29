"""LLM providers behind one function. All supported providers speak the OpenAI chat API."""
import logging
import re

import openai
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "modal": "https://hariom-nagar--ep-qwen3-8-max-vl-thinking-server.us-west.modal.direct/v1",
}


class LLMError(Exception):
    """User-presentable LLM failure (never contains keys)."""


def _client() -> OpenAI:
    provider = settings.llm_provider.lower()
    if provider == "modal":
        if not (settings.modal_proxy_token_id and settings.modal_proxy_token_secret):
            raise LLMError("Modal credentials are missing. Set MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET.")
        api_key = f"{settings.modal_proxy_token_id}.{settings.modal_proxy_token_secret}"
    elif provider == "openai":
        api_key = settings.openai_api_key or settings.llm_api_key
        if not api_key:
            raise LLMError("OPENAI_API_KEY is missing.")
    elif provider in ("groq", "gemini", "custom"):
        if not settings.llm_api_key:
            raise LLMError(f"LLM_API_KEY is missing for provider '{provider}'.")
        api_key = settings.llm_api_key
    else:
        raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'.")

    base_url = settings.llm_base_url or DEFAULT_BASE_URLS.get(provider)
    if not base_url:
        raise LLMError("LLM_BASE_URL is required for provider 'custom'.")
    return OpenAI(base_url=base_url, api_key=api_key, timeout=settings.llm_timeout_seconds, max_retries=1)


def _strip_thinking(text: str) -> str:
    # Reasoning models may inline <think>...</think>; users should only see the answer.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate(messages: list[dict]) -> str:
    client = _client()
    provider = settings.llm_provider.lower()
    params = {"model": settings.llm_model, "messages": messages}
    if provider == "openai":
        # OpenAI replaced max_tokens; reasoning models (o-series, gpt-5) only accept the default temperature
        params["max_completion_tokens"] = settings.llm_max_tokens
        if not settings.llm_model.lower().startswith(("o1", "o3", "o4", "gpt-5")):
            params["temperature"] = 0.2
    else:
        params["max_tokens"] = settings.llm_max_tokens
        params["temperature"] = 0.2
    if provider == "modal":
        params["extra_body"] = {"reasoning_effort": "high"}
    try:
        completion = client.chat.completions.create(**params)
    except openai.AuthenticationError as exc:
        raise LLMError("The AI provider rejected the credentials.") from exc
    except openai.RateLimitError as exc:
        raise LLMError("The AI provider rate limit was reached. Try again shortly.") from exc
    except openai.NotFoundError as exc:
        raise LLMError(
            f"Model '{settings.llm_model}' is not available at the provider (wrong name or deprecated). "
            "Change LLM_MODEL in backend/.env."
        ) from exc
    except openai.APITimeoutError as exc:
        raise LLMError("The AI provider timed out.") from exc
    except openai.APIConnectionError as exc:
        raise LLMError("Could not connect to the AI provider.") from exc
    except openai.APIError as exc:
        logger.warning("LLM API error: %s", type(exc).__name__)
        raise LLMError("The AI provider returned an error.") from exc

    content = completion.choices[0].message.content or ""
    answer = _strip_thinking(content)
    if not answer:
        raise LLMError("The AI provider returned an empty answer.")
    return answer
