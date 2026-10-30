import uuid
from decimal import Decimal
from sqlalchemy import ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, GUID, TimestampMixin, UUIDPrimaryKeyMixin
from src.models.enums import SceneGenerationStatus


class Scene(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "scenes"

    __table_args__ = (
        UniqueConstraint("story_id", "scene_number", name="uq_scenes_story_scene_number"),
    )

    story_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("stories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scene_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    duration_seconds: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        nullable=False,
    )
    visual_prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    narration_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    subtitle_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    generation_status: Mapped[SceneGenerationStatus] = mapped_column(
        String(50),
        default=SceneGenerationStatus.PENDING,
        nullable=False,
        index=True,
    )
    generated_asset_url: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
    )
    provider_job_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # Relationships
    story = relationship("Story", back_populates="scenes")
