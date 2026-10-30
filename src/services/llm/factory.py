from src.config import settings
from src.services.llm.base import BaseLLMProvider
from src.services.llm.mock import MockLLMProvider


def get_llm_provider() -> BaseLLMProvider:
    """Factory returning active LLM provider based on settings."""
    if settings.LLM_PROVIDER.lower() == "gemini" and settings.LLM_API_KEY:
        try:
            # If Google GenAI is configured, we can import and instantiate
            from src.services.llm.gemini import GeminiLLMProvider
            return GeminiLLMProvider(api_key=settings.LLM_API_KEY)
        except Exception:
            return MockLLMProvider()
    return MockLLMProvider()
