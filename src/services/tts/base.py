import os
import uuid
from abc import ABC, abstractmethod
from src.services.media.storage import storage_service


class TTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str, language: str) -> str:
        """
        Synthesizes text into speech.
        Returns: Absolute local path to generated WAV/MP3 file.
        """
        pass


class MockTTSProvider(TTSProvider):
    """
    Synthesizes a standard 3-second audio tone/file for development and testing.
    """

    def synthesize(self, text: str, language: str) -> str:
        audio_filename = f"tts_{uuid.uuid4().hex[:10]}.wav"
        audio_path = storage_service.get_audio_path(audio_filename)

        # Write a minimalist valid 44-byte WAV header + silence for standard media players/FFmpeg
        if not os.path.exists(audio_path):
            sample_rate = 16000
            num_channels = 1
            bits_per_sample = 16
            byte_rate = sample_rate * num_channels * bits_per_sample // 8
            block_align = num_channels * bits_per_sample // 8
            duration_sec = 3
            num_samples = sample_rate * duration_sec
            data_size = num_samples * block_align
            total_size = 36 + data_size

            header = bytearray()
            header.extend(b"RIFF")
            header.extend(total_size.to_bytes(4, "little"))
            header.extend(b"WAVE")
            header.extend(b"fmt ")
            header.extend((16).to_bytes(4, "little"))  # Subchunk1Size
            header.extend((1).to_bytes(2, "little"))   # PCM
            header.extend(num_channels.to_bytes(2, "little"))
            header.extend(sample_rate.to_bytes(4, "little"))
            header.extend(byte_rate.to_bytes(4, "little"))
            header.extend(block_align.to_bytes(2, "little"))
            header.extend(bits_per_sample.to_bytes(2, "little"))
            header.extend(b"data")
            header.extend(data_size.to_bytes(4, "little"))

            # Write silence (zeros)
            silence = bytearray(data_size)
            with open(audio_path, "wb") as f:
                f.write(header)
                f.write(silence)

        return audio_path


def get_tts_provider() -> TTSProvider:
    return MockTTSProvider()
