"""
agent/llama_client.py — Llama 3.3 70B API call handler (Together AI / OpenRouter / Groq)
"""
import httpx
import logging
from typing import Optional
from config import get_settings

logger = logging.getLogger(__name__)


async def call_llama(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> str:
    """
    Call the Llama 3.3 70B Instruct API via Together AI (or compatible endpoint).

    Args:
        system_prompt: system context with injected data
        user_message: user's natural language query
        temperature: generation temperature (lower = more factual)
        max_tokens: max response length

    Returns:
        AI response string
    """
    settings = get_settings()

    if not settings.llama_api_key:
        raise ValueError("LLAMA_API_KEY not set in .env")

    headers = {
        "Authorization": f"Bearer {settings.llama_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.llama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "stream": False,
    }

    api_url = f"{settings.llama_api_base.rstrip('/')}/v1/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        logger.error(f"Llama API HTTP error: {e.response.status_code} — {e.response.text}")
        raise RuntimeError(f"Llama API error {e.response.status_code}: {e.response.text}")
    except httpx.TimeoutException:
        raise RuntimeError("Llama API request timed out after 60 seconds")
    except Exception as e:
        logger.error(f"Llama API unexpected error: {e}")
        raise RuntimeError(f"Llama API call failed: {str(e)}")
