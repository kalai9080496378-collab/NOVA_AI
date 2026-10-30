import uuid
from datetime import datetime
from decimal import Decimal
from pydantic import Field
from src.models.enums import ModerationStatus, VideoVisibility
from src.schemas.common import BaseSchema
from src.schemas.user import UserPublicResponse


class VideoResponse(BaseSchema):
    id: uuid.UUID
    user_id: uuid.UUID
    story_id: uuid.UUID
    title: str
    description: str | None = None
    video_url: str
    thumbnail_url: str | None = None
    duration_seconds: Decimal
    visual_style: str
    ai_generated: bool
    moderation_status: ModerationStatus
    visibility: VideoVisibility
    published_at: datetime | None = None
    like_count: int
    comment_count: int
    created_at: datetime
    updated_at: datetime


class VideoUpdate(BaseSchema):
    title: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=2000)
    visibility: VideoVisibility | None = None


class VideoFeedItem(BaseSchema):
    id: uuid.UUID
    story_id: uuid.UUID
    title: str
    description: str | None = None
    video_url: str
    thumbnail_url: str | None = None
    duration_seconds: Decimal
    visual_style: str
    published_at: datetime | None = None
    like_count: int
    comment_count: int
    creator: UserPublicResponse
    is_liked_by_me: bool = False
    is_saved_by_me: bool = False


class CursorPage(BaseSchema):
    items: list[VideoFeedItem]
    next_cursor: str | None = None
    has_more: bool
