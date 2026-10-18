"""
NOVA LLM Service Module
Dedicated, configurable service for Large Language Model (LLM) providers.
Supports Google Gemini and OpenAI REST APIs with robust error handling,
free-tier explanations, API key masking, and graceful fallbacks.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import requests

from nova_core.config import (
    LLM_PROVIDER,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    is_llm_configured,
)


class LLMResponse:
    """Standardized response from any LLM provider."""
    def __init__(
        self,
        success: bool,
        content: str,
        provider: str,
        model: str,
        tokens_used: Optional[int] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ):
        self.success = success
        self.content = content
        self.provider = provider
        self.model = model
        self.tokens_used = tokens_used
        self.error = error
        self.error_type = error_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "error": self.error,
            "error_type": self.error_type,
        }

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else f"FAILED ({self.error_type})"
        return f"<LLMResponse [{self.provider}/{self.model}] {status}>"


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generates a text completion given conversation messages."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Returns provider metadata, cost information, and limitations."""
        pass


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API Provider.
    Uses the official documented Google Generative Language REST API.
    API keys are sent via HTTP headers (x-goog-api-key) rather than URLs
    to prevent credential exposure in network logs.
    """

    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
    ):
        raw_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
        self.api_key = (raw_key or "").strip()
        self.model = model or os.getenv("GEMINI_MODEL", self.DEFAULT_MODEL).strip()
        self.timeout = timeout
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                success=False,
                content="",
                provider="gemini",
                model=self.model,
                error="Google Gemini API key is missing. Set GEMINI_API_KEY in your .env file.",
                error_type="missing_credentials",
            )

        # Map role 'assistant' -> 'model' for Gemini schema
        gemini_contents = []
        for msg in messages:
            role = "model" if msg.get("role") in {"assistant", "model"} else "user"
            content_text = msg.get("content", "").strip()
            if content_text:
                gemini_contents.append({
                    "role": role,
                    "parts": [{"text": content_text}],
                })

        if not gemini_contents:
            return LLMResponse(
                success=False,
                content="",
                provider="gemini",
                model=self.model,
                error="No valid conversation messages provided.",
                error_type="empty_input",
            )

        payload: Dict[str, Any] = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": system_prompt}]
            }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            # Handle HTTP errors gracefully
            if response.status_code == 400:
                err_data = self._safe_json(response)
                msg = err_data.get("error", {}).get("message", "Invalid request format.")
                return LLMResponse(
                    success=False,
                    content="",
                    provider="gemini",
                    model=self.model,
                    error=f"Gemini API 400 (Bad Request): {msg}",
                    error_type="invalid_request",
                )
            elif response.status_code in {401, 403}:
                return LLMResponse(
                    success=False,
                    content="",
                    provider="gemini",
                    model=self.model,
                    error="Gemini API Authentication Failed: The API key provided is invalid or unauthorized.",
                    error_type="auth_error",
                )
            elif response.status_code == 429:
                return LLMResponse(
                    success=False,
                    content="",
                    provider="gemini",
                    model=self.model,
                    error="Gemini API Rate Limit Exceeded: Free tier rate limits reached (15 RPM). Please wait a moment.",
                    error_type="rate_limit",
                )
            elif response.status_code >= 500:
                return LLMResponse(
                    success=False,
                    content="",
                    provider="gemini",
                    model=self.model,
                    error=f"Google Gemini service error (HTTP {response.status_code}). Please try again shortly.",
                    error_type="server_error",
                )

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return LLMResponse(
                    success=False,
                    content="",
                    provider="gemini",
                    model=self.model,
                    error="Gemini returned no candidates (output may have triggered content filtering).",
                    error_type="empty_candidate",
                )

            parts = candidates[0].get("content", {}).get("parts", [])
            text_result = "".join(p.get("text", "") for p in parts).strip()
            tokens = data.get("usageMetadata", {}).get("totalTokenCount")

            return LLMResponse(
                success=True,
                content=text_result,
                provider="gemini",
                model=self.model,
                tokens_used=tokens,
            )

        except requests.exceptions.Timeout:
            return LLMResponse(
                success=False,
                content="",
                provider="gemini",
                model=self.model,
                error=f"Network request to Gemini timed out after {self.timeout} seconds.",
                error_type="timeout",
            )
        except requests.exceptions.ConnectionError:
            return LLMResponse(
                success=False,
                content="",
                provider="gemini",
                model=self.model,
                error="Network connection failure. Please check your internet connection.",
                error_type="network_error",
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                content="",
                provider="gemini",
                model=self.model,
                error=f"Unexpected error communicating with Gemini API: {str(e)}",
                error_type="unknown",
            )

    def get_info(self) -> Dict[str, Any]:
        return {
            "provider": "Google Gemini",
            "model": self.model,
            "cost_tier": "Free Tier Available via Google AI Studio",
            "free_tier_limits": (
                "15 Requests Per Minute (RPM), 1,000,000 Tokens Per Minute (TPM), "
                "1,500 Requests Per Day (RPD). Completely free of charge with no credit card required."
            ),
            "pricing": "Free tier for personal & prototype use. Paid pay-as-you-go optional.",
            "masked_key": _mask_key(self.api_key),
        }

    def _safe_json(self, response: requests.Response) -> Dict[str, Any]:
        try:
            return response.json()
        except Exception:
            return {}


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI-compatible Chat Completions API Provider.
    Supports OpenAI, Groq, and local Ollama or vLLM endpoints.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ):
        raw_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
        self.api_key = (raw_key or "").strip()
        self.model = model or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL).strip()
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.timeout = timeout
        self.endpoint = f"{self.base_url}/chat/completions"

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                success=False,
                content="",
                provider="openai",
                model=self.model,
                error="OpenAI API key is missing. Set OPENAI_API_KEY in your .env file.",
                error_type="missing_credentials",
            )

        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            content_text = msg.get("content", "").strip()
            if content_text:
                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "content": content_text,
                })

        if not formatted_messages:
            return LLMResponse(
                success=False,
                content="",
                provider="openai",
                model=self.model,
                error="No valid conversation messages provided.",
                error_type="empty_input",
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code in {401, 403}:
                return LLMResponse(
                    success=False,
                    content="",
                    provider="openai",
                    model=self.model,
                    error="OpenAI Authentication Failed: Invalid API key.",
                    error_type="auth_error",
                )
            elif response.status_code == 429:
                return LLMResponse(
                    success=False,
                    content="",
                    provider="openai",
                    model=self.model,
                    error="OpenAI Rate Limit or Quota Exceeded. Check your billing account balance.",
                    error_type="rate_limit",
                )
            elif response.status_code >= 400:
                err_data = response.json() if response.text else {}
                err_msg = err_data.get("error", {}).get("message", f"HTTP {response.status_code}")
                return LLMResponse(
                    success=False,
                    content="",
                    provider="openai",
                    model=self.model,
                    error=f"OpenAI API Error: {err_msg}",
                    error_type="api_error",
                )

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return LLMResponse(
                    success=False,
                    content="",
                    provider="openai",
                    model=self.model,
                    error="OpenAI returned an empty response.",
                    error_type="empty_candidate",
                )

            text_result = choices[0].get("message", {}).get("content", "").strip()
            tokens = data.get("usage", {}).get("total_tokens")

            return LLMResponse(
                success=True,
                content=text_result,
                provider="openai",
                model=self.model,
                tokens_used=tokens,
            )

        except requests.exceptions.Timeout:
            return LLMResponse(
                success=False,
                content="",
                provider="openai",
                model=self.model,
                error=f"Network request to OpenAI timed out after {self.timeout} seconds.",
                error_type="timeout",
            )
        except requests.exceptions.ConnectionError:
            return LLMResponse(
                success=False,
                content="",
                provider="openai",
                model=self.model,
                error="Network connection failure. Please check your internet connection.",
                error_type="network_error",
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                content="",
                provider="openai",
                model=self.model,
                error=f"Unexpected error communicating with OpenAI API: {str(e)}",
                error_type="unknown",
            )

    def get_info(self) -> Dict[str, Any]:
        return {
            "provider": "OpenAI",
            "model": self.model,
            "cost_tier": "Pay-as-you-go",
            "free_tier_limits": "Varies by account tier; requires prepaid account balance or API credits.",
            "pricing": "e.g., $0.15 / 1M input tokens, $0.60 / 1M output tokens for gpt-4o-mini.",
            "masked_key": _mask_key(self.api_key),
        }


def _mask_key(key: Optional[str]) -> str:
    """Masks an API key for safe display in logs and UI without leaking secrets."""
    if not key or not str(key).strip():
        return "Not Set"
    cleaned = str(key).strip()
    if len(cleaned) <= 8:
        return "****"
    return f"{cleaned[:4]}...{cleaned[-4:]}"


def get_llm_service() -> Optional[BaseLLMProvider]:
    """
    Factory function returning the active LLM provider instance,
    or None if NOVA is operating in Phase 1 Rule-Based mode.
    """
    provider_name = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()

    if provider_name == "gemini" and os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip():
        return GeminiProvider()

    if provider_name == "openai" and os.getenv("OPENAI_API_KEY", OPENAI_API_KEY).strip():
        return OpenAIProvider()

    # Automatic fallback detection if key is present even if LLM_PROVIDER wasn't explicitly changed
    if os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip():
        return GeminiProvider()

    if os.getenv("OPENAI_API_KEY", OPENAI_API_KEY).strip():
        return OpenAIProvider()

    return None
