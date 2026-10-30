from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_current_user_optional
from src.core.pagination import decode_cursor, encode_cursor
from src.database import get_db
from src.models.enums import ModerationStatus, VideoVisibility
from src.models.like import Like
from src.models.saved_video import SavedVideo
from src.models.user import User
from src.models.video import Video
from src.schemas.user import UserPublicResponse
from src.schemas.video import CursorPage, VideoFeedItem

router = APIRouter(prefix="/feed", tags=["Discovery Feed"])


@router.get(
    "",
    response_model=CursorPage,
    summary="Get paginated discovery feed of public videos",
)
async def get_discovery_feed(
    cursor: str | None = Query(None, description="Base64 encoded cursor for pagination"),
    limit: int = Query(20, ge=1, le=50),
    current_user: User | None = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
) -> CursorPage:
    """
    High-performance cursor-based pagination for public videos.
    Ordered by published_at DESC, id DESC.
    """
    query = (
        select(Video, User)
        .join(User, User.id == Video.user_id)
        .where(
            Video.visibility == VideoVisibility.PUBLIC,
            Video.moderation_status == ModerationStatus.APPROVED,
            Video.published_at.is_not(None),
        )
    )

    if cursor:
        decoded = decode_cursor(cursor)
        if decoded:
            c_published_at, c_id = decoded
            query = query.where(
                or_(
                    Video.published_at < c_published_at,
                    and_(Video.published_at == c_published_at, Video.id < c_id),
                )
            )

    query = query.order_by(Video.published_at.desc(), Video.id.desc()).limit(limit + 1)
    results = (await db.execute(query)).all()

    has_more = len(results) > limit
    items_to_return = results[:limit]

    # Pre-fetch user likes and saved state if authenticated
    liked_video_ids = set()
    saved_video_ids = set()
    if current_user and items_to_return:
        video_ids = [v.id for v, _ in items_to_return]
        liked_rows = await db.execute(
            select(Like.video_id).where(
                Like.user_id == current_user.id,
                Like.video_id.in_(video_ids),
            )
        )
        liked_video_ids = set(liked_rows.scalars().all())

        saved_rows = await db.execute(
            select(SavedVideo.video_id).where(
                SavedVideo.user_id == current_user.id,
                SavedVideo.video_id.in_(video_ids),
            )
        )
        saved_video_ids = set(saved_rows.scalars().all())

    feed_items = []
    for v, u in items_to_return:
        feed_items.append(
            VideoFeedItem(
                id=v.id,
                story_id=v.story_id,
                title=v.title,
                description=v.description,
                video_url=v.video_url,
                thumbnail_url=v.thumbnail_url,
                duration_seconds=v.duration_seconds,
                visual_style=v.visual_style,
                published_at=v.published_at,
                like_count=v.like_count,
                comment_count=v.comment_count,
                creator=UserPublicResponse.model_validate(u),
                is_liked_by_me=v.id in liked_video_ids,
                is_saved_by_me=v.id in saved_video_ids,
            )
        )

    next_cursor = None
    if has_more and items_to_return:
        last_v, _ = items_to_return[-1]
        next_cursor = encode_cursor(last_v.published_at, last_v.id)

    return CursorPage(
        items=feed_items,
        next_cursor=next_cursor,
        has_more=has_more,
    )
