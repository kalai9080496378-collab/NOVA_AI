from datetime import timedelta
from decimal import Decimal


def format_srt_time(seconds: float) -> str:
    """Formats float seconds into standard SRT format: HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis = int((seconds - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_subtitles(scenes: list, output_filepath: str) -> str:
    """
    Generates an SRT subtitle file from scenes using their duration_seconds and subtitle_text.
    Preserves Thanglish Roman script as-is to avoid complex Tamil HarfBuzz shaping issues.
    """
    srt_lines = []
    current_time = 0.0

    for idx, scene in enumerate(scenes, start=1):
        dur = float(scene.duration_seconds)
        start_time = current_time
        end_time = current_time + dur
        current_time = end_time

        text = (scene.subtitle_text or scene.narration_text or "").strip()
        if not text:
            continue

        srt_lines.append(f"{idx}")
        srt_lines.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
        srt_lines.append(text)
        srt_lines.append("")  # Blank line delimiter

    content = "\n".join(srt_lines)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return output_filepath
