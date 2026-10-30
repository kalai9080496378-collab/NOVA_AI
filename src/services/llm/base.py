from abc import ABC, abstractmethod
import uuid
from pydantic import BaseModel, Field


class LLMSceneItem(BaseModel):
    scene_number: int = Field(..., ge=1)
    duration: float = Field(..., gt=0)
    visual_prompt: str = Field(...)
    narration_text: str = Field(...)
    subtitle_text: str = Field(...)


class LLMScenePlanOutput(BaseModel):
    scenes: list[LLMSceneItem]


class StoryContext(BaseModel):
    story_id: uuid.UUID
    title: str
    original_story: str
    input_language: str
    selected_style: str
    requested_duration: int


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_scenes(self, ctx: StoryContext) -> LLMScenePlanOutput:
        pass

