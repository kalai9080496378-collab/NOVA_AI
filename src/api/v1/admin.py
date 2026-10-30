import uuid
from datetime import datetime, timezone
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_current_admin_user
from src.database import get_db
from src.models.enums import ModerationStatus, ReportStatus, VideoVisibility
from src.models.report import Report
from src.models.user import User
from src.models.video import Video
from src.schemas.social import ReportResponse, ReportUpdate
from src.schemas.video import VideoResponse

router = APIRouter(prefix="/admin", tags=["Admin Moderation"])


class VideoReviewRequest(BaseModel):
    action: Literal["approve", "reject"]
    reason: str | None = None


@router.get(
    "/moderation/videos",
    response_model=list[VideoResponse],
    summary="List pending videos for content moderation review",
)
async def list_pending_videos(
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> list[Video]:
    """Admin-only: lists videos awaiting moderation approval."""
    stmt = (
        select(Video)
        .where(Video.moderation_status == ModerationStatus.PENDING)
        .order_by(Video.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    return (await db.execute(stmt)).scalars().all()


@router.post(
    "/moderation/videos/{id}/review",
    response_model=VideoResponse,
    summary="Approve or reject a video for moderation",
)
async def review_video(
    id: uuid.UUID,
    payload: VideoReviewRequest,
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> Video:
    """Admin-only: sets moderation status to approved or rejected."""
    stmt = select(Video).where(Video.id == id)
    video = (await db.execute(stmt)).scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    if payload.action == "approve":
        video.moderation_status = ModerationStatus.APPROVED
    else:
        video.moderation_status = ModerationStatus.REJECTED
        video.visibility = VideoVisibility.PRIVATE

    await db.commit()
    await db.refresh(video)
    return video


@router.get(
    "/reports",
    response_model=list[ReportResponse],
    summary="List community content reports",
)
async def list_reports(
    status_filter: ReportStatus | None = None,
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> list[Report]:
    """Admin-only: lists reports submitted by community users."""
    query = select(Report)
    if status_filter:
        query = query.where(Report.status == status_filter)
    query = query.order_by(Report.created_at.desc()).limit(limit).offset(offset)
    return (await db.execute(query)).scalars().all()


@router.patch(
    "/reports/{id}",
    response_model=ReportResponse,
    summary="Update report review status",
)
async def update_report(
    id: uuid.UUID,
    payload: ReportUpdate,
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> Report:
    """Admin-only: resolves or dismisses a report."""
    stmt = select(Report).where(Report.id == id)
    report = (await db.execute(stmt)).scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    report.status = payload.status
    report.reviewed_by = current_admin.id
    report.reviewed_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(report)
    return report
