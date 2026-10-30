from abc import ABC, abstractmethod
from src.services.llm.base import LLMSceneItem, LLMScenePlanOutput, StoryContext
from src.services.llm.prompts import build_scene_planning_prompt


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_scenes(self, ctx: StoryContext) -> LLMScenePlanOutput:
        pass


class MockLLMProvider(BaseLLMProvider):
    """
    Deterministic mock provider generating compliant scenes with exact duration sums,
    style hints, and authentic Thanglish preservation for local dev and tests.
    """
    async def generate_scenes(self, ctx: StoryContext) -> LLMScenePlanOutput:
        count = {10: 2, 15: 3, 20: 4, 30: 5}.get(ctx.requested_duration, 3)
        base_dur = round(ctx.requested_duration / count, 2)
        scenes = []

        style_hint = f"styled with {ctx.selected_style} aesthetic"

        for i in range(1, count + 1):
            dur = base_dur if i < count else round(ctx.requested_duration - (base_dur * (count - 1)), 2)

            if ctx.input_language == "thanglish":
                narration = f"Scene {i} la namma hero enter aaguraru with full style and energy."
                subtitle = f"Scene {i} la namma hero enter aaguraru with full style and energy."
            elif ctx.input_language == "ta":
                narration = f"காட்சி {i}: கதாநாயகன் கம்பீரமாக நுழைகிறான்."
                subtitle = f"காட்சி {i}: கதாநாயகன் கம்பீரமாக நுழைகிறான்."
            else:
                narration = f"Scene {i}: The protagonist embarks on an unexpected journey."
                subtitle = f"Scene {i}: The protagonist embarks on an unexpected journey."

            scenes.append(
                LLMSceneItem(
                    scene_number=i,
                    duration=dur,
                    visual_prompt=f"A cinematic view of scene {i} for '{ctx.title}', {style_hint}, dramatic lighting, 8k resolution.",
                    narration_text=narration,
                    subtitle_text=subtitle,
                )
            )

        return LLMScenePlanOutput(scenes=scenes)
