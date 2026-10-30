import uuid
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.models.base import Base, GUID, TimestampMixin, UUIDPrimaryKeyMixin


class SavedVideo(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "saved_videos"

    __table_args__ = (
        UniqueConstraint("user_id", "video_id", name="uq_saved_videos_user_video"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Relationships
    user = relationship("User", back_populates="saved_videos")
    video = relationship("Video", back_populates="saved_by")
