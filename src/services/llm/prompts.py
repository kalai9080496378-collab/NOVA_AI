def build_scene_planning_prompt(
    title: str,
    original_story: str,
    input_language: str,
    selected_style: str,
    requested_duration: int,
) -> str:
    """Constructs the prompt for LLM scene planning with style hints and Thanglish preservation."""
    style_guidance = {
        "realistic": "Hyper-realistic cinematic cinematography, 8k resolution, documentary lighting, photoreal film grain.",
        "cartoon_3d": "Stylized 3D digital animation, Pixar/DreamWorks aesthetic, vibrant volumetric lighting, soft bevels.",
        "anime": "Japanese anime aesthetic, Makoto Shinkai style, hand-drawn keyframes, vivid cel shading, cinematic depth of field.",
    }.get(selected_style.lower(), "High quality cinematic visual aesthetic.")

    thanglish_rule = ""
    if input_language.lower() == "thanglish":
        thanglish_rule = (
            "\nCRITICAL INSTRUCTION FOR THANGLISH:\n"
            "The user's story is written in Thanglish (Tamil words in Roman script mixed with English). "
            "Generate narration_text in the same style. Do NOT normalize to standard Tamil or English. "
            "Keep the authentic code-mixed Romanized conversational phrasing in narration_text and subtitle_text.\n"
        )

    # Determine recommended scene counts
    scene_count_map = {10: 2, 15: 3, 20: 4, 30: 5}
    num_scenes = scene_count_map.get(requested_duration, 3)

    return f"""You are the lead cinematic AI director for NOVA AI, an automated story-to-video studio.
Break down the following story into exactly {num_scenes} sequential scenes for a short video.

STORY TITLE: {title}
STORY TEXT: {original_story}
INPUT LANGUAGE: {input_language}
SELECTED VISUAL STYLE: {selected_style}
REQUESTED DURATION: {requested_duration} seconds
{thanglish_rule}
REQUIREMENTS:
1. Return strictly JSON with the key "scenes".
2. The duration of each scene must be a positive float.
3. The sum of all scene durations MUST equal exactly {requested_duration} seconds.
4. Each scene's visual_prompt MUST include style hints matching "{selected_style}": {style_guidance}
5. narration_text: Spoken narrative for this scene.
6. subtitle_text: On-screen caption text corresponding to narration_text.

FORMAT:
{{
  "scenes": [
    {{
      "scene_number": 1,
      "duration": 5.0,
      "visual_prompt": "Cinematic visual description with {selected_style} styling...",
      "narration_text": "Spoken line...",
      "subtitle_text": "Caption text..."
    }}
  ]
}}
"""
