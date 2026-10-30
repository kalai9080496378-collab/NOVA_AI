from src.config import settings
from src.services.video.base import VideoGenerationProvider
from src.services.video.mock import MockVideoProvider
from src.services.video.runapi_veo import RunAPIVeoProvider


def get_video_provider() -> VideoGenerationProvider:
    """Factory resolving video generation provider based on settings."""
    provider_type = settings.VIDEO_PROVIDER.lower()
    if provider_type == "runapi" and settings.RUNAPI_API_KEY:
        try:
            return RunAPIVeoProvider(api_key=settings.RUNAPI_API_KEY)
        except Exception:
            return MockVideoProvider()
    return MockVideoProvider()
