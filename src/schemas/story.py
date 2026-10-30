import uuid
from datetime import datetime
from decimal import Decimal
from typing import Literal
from pydantic import Field
from src.models.enums import InputLanguage, SceneGenerationStatus, StoryStatus, VisualStyle
from src.schemas.common import BaseSchema


class SceneItem(BaseSchema):
    id: uuid.UUID
    scene_number: int
    duration_seconds: Decimal
    visual_prompt: str
    narration_text: str | None = None
    subtitle_text: str | None = None
    generation_status: SceneGenerationStatus
    generated_asset_url: str | None = None
    provider_job_id: str | None = None


class StoryCreate(BaseSchema):
    title: str = Field(..., min_length=1, max_length=255)
    original_story: str = Field(..., min_length=10, max_length=10000)
    input_language: InputLanguage = Field(
        ...,
        description="Explicit selection: en, ta, or thanglish (never auto-detected)",
    )
    selected_style: VisualStyle = Field(
        ...,
        description="Style: realistic, cartoon_3d, or anime",
    )
    requested_duration: Literal[10, 15, 20, 30] = Field(
        ...,
        description="Target duration must be 10, 15, 20, or 30 seconds",
    )


class StoryResponse(BaseSchema):
    id: uuid.UUID
    user_id: uuid.UUID
    title: str
    original_story: str
    input_language: InputLanguage
    selected_style: VisualStyle
    requested_duration: int
    status: StoryStatus
    created_at: datetime
    updated_at: datetime
    scenes: list[SceneItem] = []


class StoryStatusResponse(BaseSchema):
    story_id: uuid.UUID
    title: str
    status: StoryStatus
    stage: str
    progress_percent: int
    scenes_count: int
    completed_scenes: int
    video_id: uuid.UUID | None = None
    video_url: str | None = None
    error: str | None = None


class SceneStatusResponse(BaseSchema):
    scene_id: uuid.UUID
    story_id: uuid.UUID
    scene_number: int
    generation_status: SceneGenerationStatus
    provider_job_id: str | None = None
    generated_asset_url: str | None = None
    duration_seconds: Decimal
    created_at: datetime
    updated_at: datetime
