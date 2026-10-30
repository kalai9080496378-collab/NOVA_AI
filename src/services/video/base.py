from abc import ABC, abstractmethod


class VideoGenerationProvider(ABC):
    @abstractmethod
    def generate_clip(
        self,
        prompt: str,
        duration: float,
        style: str,
        aspect_ratio: str = "16:9",
    ) -> dict:
        """
        Submits video generation job to provider.
        Returns: {"job_id": str, "status": "processing"}
        """
        pass

    @abstractmethod
    def get_status(self, job_id: str) -> dict:
        """
        Polls job status from provider.
        Returns: {
            "status": "processing" | "completed" | "failed",
            "video_url": str | None,
            "error": str | None
        }
        """
        pass
