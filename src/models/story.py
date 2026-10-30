import uuid
from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, GUID, TimestampMixin, UUIDPrimaryKeyMixin
from src.models.enums import InputLanguage, StoryStatus, VisualStyle


class Story(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "stories"

    __table_args__ = (
        CheckConstraint(
            "requested_duration IN (10, 15, 20, 30)",
            name="ck_stories_requested_duration",
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    original_story: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    input_language: Mapped[InputLanguage] = mapped_column(
        String(20),
        nullable=False,
    )
    selected_style: Mapped[VisualStyle] = mapped_column(
        String(50),
        nullable=False,
    )
    requested_duration: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    status: Mapped[StoryStatus] = mapped_column(
        String(50),
        default=StoryStatus.STORY_SUBMITTED,
        nullable=False,
        index=True,
    )

    # Relationships
    user = relationship("User", back_populates="stories")
    scenes = relationship(
        "Scene",
        back_populates="story",
        cascade="all, delete-orphan",
        order_by="Scene.scene_number",
        lazy="selectin",
    )
    video = relationship(
        "Video",
        back_populates="story",
        uselist=False,
        cascade="all, delete-orphan",
    )
    generation_jobs = relationship(
        "GenerationJob",
        back_populates="story",
        cascade="all, delete-orphan",
    )
