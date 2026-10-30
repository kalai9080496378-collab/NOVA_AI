from datetime import datetime
from sqlalchemy import Boolean, CheckConstraint, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from src.models.enums import AccountStatus


class User(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "users"

    __table_args__ = (
        CheckConstraint(
            "length(username) >= 3",
            name="ck_users_username_format",
        ),
        CheckConstraint(
            "monthly_generation_quota >= 0",
            name="ck_users_monthly_quota_non_negative",
        ),
        CheckConstraint(
            "generation_quota_remaining >= 0",
            name="ck_users_quota_remaining_non_negative",
        ),
    )

    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    display_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    bio: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    avatar_url: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )
    account_status: Mapped[AccountStatus] = mapped_column(
        String(50),
        default=AccountStatus.ACTIVE,
        nullable=False,
    )
    is_admin: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    # Cost protection monthly quotas
    monthly_generation_quota: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
    )
    generation_quota_remaining: Mapped[int] = mapped_column(
        Integer,
        default=3,
        nullable=False,
    )
    quota_reset_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=True,
    )

    # Relationships
    stories = relationship("Story", back_populates="user", cascade="all, delete-orphan")
    videos = relationship("Video", back_populates="user", cascade="all, delete-orphan")
    likes = relationship("Like", back_populates="user", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="user", cascade="all, delete-orphan")
    saved_videos = relationship("SavedVideo", back_populates="user", cascade="all, delete-orphan")
