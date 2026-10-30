import uuid
from datetime import datetime
from decimal import Decimal
from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, GUID, TimestampMixin, UUIDPrimaryKeyMixin
from src.models.enums import ModerationStatus, VideoVisibility, VisualStyle


class Video(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "videos"

    __table_args__ = (
        CheckConstraint("like_count >= 0", name="ck_videos_like_count_non_negative"),
        CheckConstraint("comment_count >= 0", name="ck_videos_comment_count_non_negative"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    story_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("stories.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    video_url: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )
    thumbnail_url: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
    )
    duration_seconds: Mapped[Decimal] = mapped_column(
        Numeric(6, 2),
        nullable=False,
    )
    visual_style: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    ai_generated: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    moderation_status: Mapped[ModerationStatus] = mapped_column(
        String(50),
        default=ModerationStatus.PENDING,
        nullable=False,
        index=True,
    )
    visibility: Mapped[VideoVisibility] = mapped_column(
        String(50),
        default=VideoVisibility.PRIVATE,
        nullable=False,
        index=True,
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    # Denormalized engagement counters
    like_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    comment_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="videos")
    story = relationship("Story", back_populates="video")
    likes = relationship("Like", back_populates="video", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="video", cascade="all, delete-orphan")
    saved_by = relationship("SavedVideo", back_populates="video", cascade="all, delete-orphan")
