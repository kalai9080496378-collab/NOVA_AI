import os
import shutil
import uuid
import httpx
from src.config import settings


class AssetStorageService:
    """Manages persistent media assets, downloading remote clips before expiration."""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or settings.STORAGE_DIR
        self.clips_dir = os.path.join(self.base_dir, "clips")
        self.audio_dir = os.path.join(self.base_dir, "audio")
        self.subtitles_dir = os.path.join(self.base_dir, "subtitles")
        self.videos_dir = os.path.join(self.base_dir, "videos")

        for d in [self.clips_dir, self.audio_dir, self.subtitles_dir, self.videos_dir]:
            os.makedirs(d, exist_ok=True)

    def persist_from_url(self, remote_url: str, filename: str) -> str:
        """
        Downloads a remote clip (e.g. RunAPI 7-day expiring URL) to permanent local storage.
        Returns the permanent local file path.
        """
        target_path = os.path.join(self.clips_dir, filename)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return target_path

        # If it's a local file or test asset
        if remote_url.startswith("file://") or os.path.exists(remote_url):
            src_path = remote_url.replace("file://", "")
            shutil.copyfile(src_path, target_path)
            return target_path

        # Download via HTTP
        with httpx.Client(timeout=60.0) as client:
            resp = client.get(remote_url, follow_redirects=True)
            resp.raise_for_status()
            with open(target_path, "wb") as f:
                f.write(resp.content)

        return target_path

    def get_clip_path(self, filename: str) -> str:
        return os.path.join(self.clips_dir, filename)

    def get_video_path(self, filename: str) -> str:
        return os.path.join(self.videos_dir, filename)

    def get_audio_path(self, filename: str) -> str:
        return os.path.join(self.audio_dir, filename)

    def get_subtitles_path(self, filename: str) -> str:
        return os.path.join(self.subtitles_dir, filename)


storage_service = AssetStorageService()
