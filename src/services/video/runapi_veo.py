import time
from src.services.media.storage import storage_service
from src.services.video.base import VideoGenerationProvider


class RunAPIVeoProvider(VideoGenerationProvider):
    """
    Real video provider using RunAPI Veo 3.1 (runapi-veo-3-1 package).
    Features:
    - Calls client.text_to_video.create(model="veo-3.1-fast", prompt=..., duration_seconds=8)
    - 2-stage retry logic on timeouts or rate limit errors
    - Immediate local asset ingestion to prevent loss from RunAPI 7-day URL expiration
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            from runapi_veo_3_1 import RunAPI
            self.client = RunAPI(api_key=api_key)
        except Exception:
            self.client = None

    def _execute_with_retry(self, fn, max_retries: int = 2):
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_err = e
                # Check for rate limit or network timeout
                time.sleep(2 ** attempt)
        raise last_err

    def generate_clip(
        self,
        prompt: str,
        duration: float,
        style: str,
        aspect_ratio: str = "16:9",
    ) -> dict:
        if not self.client:
            raise RuntimeError("RunAPI client not initialized. Check RUNAPI_API_KEY.")

        enhanced_prompt = f"{prompt}, visual style: {style}, {aspect_ratio} aspect ratio"
        duration_sec = 8 if duration >= 6.0 else 5

        task = self._execute_with_retry(
            lambda: self.client.text_to_video.create(
                model="veo-3.1-fast",
                prompt=enhanced_prompt,
                duration_seconds=duration_sec,
            )
        )
        return {"job_id": str(task.id), "status": "processing"}

    def get_status(self, job_id: str) -> dict:
        if not self.client:
            raise RuntimeError("RunAPI client not initialized.")

        task = self._execute_with_retry(lambda: self.client.text_to_video.get(job_id))
        status = getattr(task, "status", "processing")

        if status == "completed":
            remote_url = getattr(task, "video_url", None) or getattr(task, "url", None)
            if not remote_url:
                return {"status": "failed", "video_url": None, "error": "No video URL in completed task"}

            # CRITICAL: Ingest immediately before RunAPI 7-day expiration!
            filename = f"runapi_{job_id}.mp4"
            permanent_local_path = storage_service.persist_from_url(remote_url, filename)
            return {
                "status": "completed",
                "video_url": permanent_local_path,
                "error": None,
            }
        elif status == "failed":
            error_msg = getattr(task, "error", "Generation failed on provider")
            return {"status": "failed", "video_url": None, "error": str(error_msg)}

        return {"status": "processing", "video_url": None, "error": None}
