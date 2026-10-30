import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_current_user
from src.database import get_db
from src.models.comment import Comment
from src.models.enums import ReportStatus, TargetType
from src.models.report import Report
from src.models.user import User
from src.models.video import Video
from src.schemas.social import ReportCreate, ReportResponse

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.post(
    "",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Report inappropriate content or user",
)
async def submit_report(
    payload: ReportCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Report:
    """Submits a content report for administrative review."""
    # Verify target entity exists
    if payload.target_type == TargetType.VIDEO:
        target = (await db.execute(select(Video).where(Video.id == payload.target_id))).scalar_one_or_none()
    elif payload.target_type == TargetType.COMMENT:
        target = (await db.execute(select(Comment).where(Comment.id == payload.target_id))).scalar_one_or_none()
    else:
        target = (await db.execute(select(User).where(User.id == payload.target_id))).scalar_one_or_none()

    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reported {payload.target_type.value} target entity does not exist.",
        )

    report = Report(
        reporter_id=current_user.id,
        target_type=payload.target_type,
        target_id=payload.target_id,
        reason=payload.reason.strip(),
        status=ReportStatus.PENDING,
    )
    db.add(report)
    await db.commit()
    await db.refresh(report)
    return report
