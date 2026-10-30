import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_current_user
from src.database import get_db
from src.models.comment import Comment
from src.models.enums import ModerationStatus, VideoVisibility
from src.models.like import Like
from src.models.saved_video import SavedVideo
from src.models.story import Story
from src.models.user import User
from src.models.video import Video
from src.schemas.social import (
    CommentCreate,
    CommentResponse,
    LikeResponse,
    SaveResponse,
)
from src.schemas.user import UserPublicResponse
from src.schemas.video import VideoResponse, VideoUpdate

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.get(
    "/{id}",
    response_model=VideoResponse,
    summary="Get video details by ID",
)
async def get_video(
    id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> Video:
    """Returns video record by ID."""
    stmt = select(Video).where(Video.id == id)
    video = (await db.execute(stmt)).scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return video


@router.patch(
    "/{id}",
    response_model=VideoResponse,
    summary="Update and publish video (Strict AI-Only and Moderation Enforcement)",
)
async def update_and_publish_video(
    id: uuid.UUID,
    payload: VideoUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Video:
    """
    Publish flow and metadata update.
    Enforces AI-Only Publishing:
    1. Video must have a non-null story_id.
    2. Video must have ai_generated = True.
    3. Moderation status must be 'approved' before visibility can be set to 'public'.
    """
    stmt = select(Video).where(Video.id == id)
    video = (await db.execute(stmt)).scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    if video.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    # AI-Only checks
    if not video.story_id or not video.ai_generated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only AI-generated videos created via NOVA AI stories can be published.",
        )

    # Check public publishing requirements
    if payload.visibility == VideoVisibility.PUBLIC:
        if video.moderation_status != ModerationStatus.APPROVED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Video cannot be published while moderation status is '{video.moderation_status.value}'. Must be 'approved'.",
            )
        video.visibility = VideoVisibility.PUBLIC
        if not video.published_at:
            video.published_at = datetime.now(timezone.utc)
    elif payload.visibility:
        video.visibility = payload.visibility

    if payload.title is not None:
        video.title = payload.title
    if payload.description is not None:
        video.description = payload.description

    await db.commit()
    await db.refresh(video)
    return video


@router.post(
    "/{id}/like",
    response_model=LikeResponse,
    summary="Like a video (Self-like blocked)",
)
async def like_video(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LikeResponse:
    """Likes a video and atomically increments like_count. Users cannot like their own video."""
    video = (await db.execute(select(Video).where(Video.id == id))).scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    # Policy decision: Block liking own video
    if video.user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Users cannot like their own video.",
        )

    # Idempotent insert
    like_stmt = select(Like).where(Like.user_id == current_user.id, Like.video_id == id)
    existing_like = (await db.execute(like_stmt)).scalar_one_or_none()
    if not existing_like:
        db.add(Like(user_id=current_user.id, video_id=id))
        video.like_count += 1
        await db.commit()
        await db.refresh(video)

    return LikeResponse(video_id=video.id, liked=True, like_count=video.like_count)


@router.delete(
    "/{id}/like",
    response_model=LikeResponse,
    summary="Unlike a video",
)
async def unlike_video(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LikeResponse:
    """Removes a like and atomically decrements like_count."""
    video = (await db.execute(select(Video).where(Video.id == id))).scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    del_stmt = delete(Like).where(Like.user_id == current_user.id, Like.video_id == id)
    res = await db.execute(del_stmt)
    if res.rowcount > 0:
        video.like_count = max(video.like_count - 1, 0)
        await db.commit()
        await db.refresh(video)

    return LikeResponse(video_id=video.id, liked=False, like_count=video.like_count)


@router.post(
    "/{id}/comments",
    response_model=CommentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Post a comment (Max 500 chars)",
)
async def post_comment(
    id: uuid.UUID,
    payload: CommentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> CommentResponse:
    """Adds a comment, verifies length <= 500 chars, and increments comment_count."""
    video = (await db.execute(select(Video).where(Video.id == id))).scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    if payload.parent_comment_id:
        parent = (
            await db.execute(select(Comment).where(Comment.id == payload.parent_comment_id))
        ).scalar_one_or_none()
        if not parent or parent.video_id != video.id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid parent comment")

    comment = Comment(
        user_id=current_user.id,
        video_id=video.id,
        parent_comment_id=payload.parent_comment_id,
        content=payload.content.strip(),
    )
    db.add(comment)
    video.comment_count += 1
    await db.commit()
    await db.refresh(comment)

    return CommentResponse(
        id=comment.id,
        user_id=comment.user_id,
        video_id=comment.video_id,
        parent_comment_id=comment.parent_comment_id,
        content=comment.content,
        deleted_at=comment.deleted_at,
        created_at=comment.created_at,
        updated_at=comment.updated_at,
        user=UserPublicResponse.model_validate(current_user),
    )


@router.get(
    "/{id}/comments",
    response_model=list[CommentResponse],
    summary="Get paginated comments for a video",
)
async def get_comments(
    id: uuid.UUID,
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> list[CommentResponse]:
    """Returns comments for a video ordered by created_at DESC."""
    stmt = (
        select(Comment, User)
        .join(User, User.id == Comment.user_id)
        .where(Comment.video_id == id, Comment.deleted_at.is_(None))
        .order_by(Comment.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    results = (await db.execute(stmt)).all()

    return [
        CommentResponse(
            id=c.id,
            user_id=c.user_id,
            video_id=c.video_id,
            parent_comment_id=c.parent_comment_id,
            content=c.content,
            deleted_at=c.deleted_at,
            created_at=c.created_at,
            updated_at=c.updated_at,
            user=UserPublicResponse.model_validate(u),
        )
        for c, u in results
    ]


@router.post(
    "/{id}/save",
    response_model=SaveResponse,
    summary="Bookmark/save a video",
)
async def save_video(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SaveResponse:
    """Bookmarks a video for the current user."""
    video = (await db.execute(select(Video).where(Video.id == id))).scalar_one_or_none()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")

    check_stmt = select(SavedVideo).where(
        SavedVideo.user_id == current_user.id,
        SavedVideo.video_id == id,
    )
    existing = (await db.execute(check_stmt)).scalar_one_or_none()
    if not existing:
        db.add(SavedVideo(user_id=current_user.id, video_id=id))
        await db.commit()

    return SaveResponse(video_id=video.id, saved=True)


@router.delete(
    "/{id}/save",
    response_model=SaveResponse,
    summary="Remove bookmark from a video",
)
async def unsave_video(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SaveResponse:
    """Removes a bookmark."""
    del_stmt = delete(SavedVideo).where(
        SavedVideo.user_id == current_user.id,
        SavedVideo.video_id == id,
    )
    await db.execute(del_stmt)
    await db.commit()

    return SaveResponse(video_id=id, saved=False)
