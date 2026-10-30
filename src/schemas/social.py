import uuid
from datetime import datetime
from pydantic import Field
from src.models.enums import ReportStatus, TargetType
from src.schemas.common import BaseSchema
from src.schemas.user import UserPublicResponse


class CommentCreate(BaseSchema):
    content: str = Field(..., min_length=1, max_length=500)
    parent_comment_id: uuid.UUID | None = None


class CommentResponse(BaseSchema):
    id: uuid.UUID
    user_id: uuid.UUID
    video_id: uuid.UUID
    parent_comment_id: uuid.UUID | None = None
    content: str
    deleted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    user: UserPublicResponse


class LikeResponse(BaseSchema):
    video_id: uuid.UUID
    liked: bool
    like_count: int


class FollowResponse(BaseSchema):
    username: str
    following: bool


class SaveResponse(BaseSchema):
    video_id: uuid.UUID
    saved: bool


class ReportCreate(BaseSchema):
    target_type: TargetType
    target_id: uuid.UUID
    reason: str = Field(..., min_length=3, max_length=500)


class ReportResponse(BaseSchema):
    id: uuid.UUID
    reporter_id: uuid.UUID
    target_type: TargetType
    target_id: uuid.UUID
    reason: str
    status: ReportStatus
    reviewed_by: uuid.UUID | None = None
    reviewed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class ReportUpdate(BaseSchema):
    status: ReportStatus
    action_notes: str | None = None
