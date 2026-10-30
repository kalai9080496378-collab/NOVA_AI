import os
import time
import uuid
from src.services.media.storage import storage_service
from src.services.video.base import VideoGenerationProvider

# In-memory store of mock video jobs: { job_id: { "start_time": float, "target_seconds": float, "prompt": str } }
_mock_jobs: dict[str, dict] = {}


class MockVideoProvider(VideoGenerationProvider):
    """
    Simulates video generation over 10-30 seconds (configurable, default 10s).
    Returns a clearly labelled sample video asset path.
    """

    def __init__(self, simulation_latency: float = 10.0):
        self.simulation_latency = simulation_latency

    def generate_clip(
        self,
        prompt: str,
        duration: float,
        style: str,
        aspect_ratio: str = "16:9",
    ) -> dict:
        job_id = f"mock_veo_{uuid.uuid4().hex[:12]}"
        _mock_jobs[job_id] = {
            "start_time": time.time(),
            "target_seconds": self.simulation_latency,
            "prompt": prompt,
            "duration": duration,
        }
        return {"job_id": job_id, "status": "processing"}

    def get_status(self, job_id: str) -> dict:
        job = _mock_jobs.get(job_id)
        if not job:
            return {"status": "failed", "video_url": None, "error": "Job not found"}

        elapsed = time.time() - job["start_time"]
        if elapsed < job["target_seconds"]:
            return {"status": "processing", "video_url": None, "error": None}

        # Create dummy sample mp4 in storage if not exists
        sample_filename = f"{job_id}.mp4"
        sample_path = storage_service.get_clip_path(sample_filename)
        if not os.path.exists(sample_path):
            with open(sample_path, "wb") as f:
                f.write(b"NOVA_AI_MOCK_VIDEO_STREAM_BYTES_FOR_DEV_PREVIEW\n")

        return {
            "status": "completed",
            "video_url": sample_path,
            "error": None,
        }
