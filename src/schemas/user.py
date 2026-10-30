import uuid
from datetime import datetime
from pydantic import EmailStr, Field
from src.models.enums import AccountStatus
from src.schemas.common import BaseSchema


class UserResponse(BaseSchema):
    id: uuid.UUID
    username: str
    email: EmailStr
    display_name: str | None = None
    bio: str | None = None
    avatar_url: str | None = None
    account_status: AccountStatus
    is_admin: bool
    monthly_generation_quota: int
    generation_quota_remaining: int
    quota_reset_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    # Note: password_hash is strictly never exposed


class UserPublicResponse(BaseSchema):
    id: uuid.UUID
    username: str
    display_name: str | None = None
    bio: str | None = None
    avatar_url: str | None = None
    created_at: datetime


class UserUpdate(BaseSchema):
    display_name: str | None = Field(None, max_length=100)
    bio: str | None = Field(None, max_length=1000)
    avatar_url: str | None = Field(None, max_length=512)
