from src.services.video.base import VideoGenerationProvider


class RunwayProvider(VideoGenerationProvider):
    """
    Placeholder implementation for Runway Gen-3 Alpha Turbo API.
    TODO: Integrate Runway SDK/REST API when credentials are provided.
    """

    def generate_clip(self, prompt: str, duration: float, style: str, aspect_ratio: str = "16:9") -> dict:
        # TODO: Implement Runway Gen-3 text_to_video task submission
        raise NotImplementedError("Runway provider is a placeholder. Use RunAPI or Mock.")

    def get_status(self, job_id: str) -> dict:
        # TODO: Implement Runway Gen-3 polling
        raise NotImplementedError("Runway provider is a placeholder. Use RunAPI or Mock.")


class VeoProvider(VideoGenerationProvider):
    """
    Placeholder implementation for Google Veo API (direct GCP Vertex AI endpoint).
    TODO: Integrate Vertex AI Veo endpoint when GCP project credentials are configured.
    """

    def generate_clip(self, prompt: str, duration: float, style: str, aspect_ratio: str = "16:9") -> dict:
        # TODO: Implement Google Veo Vertex AI call
        raise NotImplementedError("Google Veo direct provider is a placeholder. Use RunAPI or Mock.")

    def get_status(self, job_id: str) -> dict:
        # TODO: Implement Google Veo polling
        raise NotImplementedError("Google Veo direct provider is a placeholder. Use RunAPI or Mock.")
