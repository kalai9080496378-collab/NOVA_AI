from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_current_user
from src.database import get_db
from src.models.enums import AccountStatus
from src.models.follow import Follow
from src.models.saved_video import SavedVideo
from src.models.user import User
from src.models.video import Video
from src.schemas.social import FollowResponse
from src.schemas.user import UserPublicResponse, UserResponse, UserUpdate
from src.schemas.video import VideoFeedItem

router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current authenticated user profile",
)
async def get_my_profile(
    current_user: User = Depends(get_current_user),
) -> User:
    """Returns authenticated user's private profile including quotas and status."""
    return current_user


@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update current authenticated user profile",
)
async def update_my_profile(
    payload: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Updates only display_name, bio, and avatar_url for the current user."""
    if payload.display_name is not None:
        current_user.display_name = payload.display_name
    if payload.bio is not None:
        current_user.bio = payload.bio
    if payload.avatar_url is not None:
        current_user.avatar_url = payload.avatar_url

    await db.commit()
    await db.refresh(current_user)
    return current_user


@router.get(
    "/me/saved",
    response_model=list[VideoFeedItem],
    summary="Get current user's bookmarked videos",
)
async def get_my_saved_videos(
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[VideoFeedItem]:
    """Returns paginated list of videos bookmarked by current user."""
    stmt = (
        select(Video)
        .join(SavedVideo, SavedVideo.video_id == Video.id)
        .where(SavedVideo.user_id == current_user.id)
        .order_by(SavedVideo.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    videos = (await db.execute(stmt)).scalars().all()

    items = []
    for v in videos:
        # Load creator
        creator = (await db.execute(select(User).where(User.id == v.user_id))).scalar_one()
        items.append(
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
                creator=UserPublicResponse.model_validate(creator),
                is_liked_by_me=False,
                is_saved_by_me=True,
            )
        )
    return items


@router.get(
    "/{username}",
    response_model=UserPublicResponse,
    summary="Get public user profile",
)
async def get_public_profile(
    username: str,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Returns non-sensitive public profile of an active user."""
    stmt = select(User).where(
        User.username == username.lower(),
        User.account_status == AccountStatus.ACTIVE,
    )
    user = (await db.execute(stmt)).scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.post(
    "/{username}/follow",
    response_model=FollowResponse,
    summary="Follow a user",
)
async def follow_user(
    username: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FollowResponse:
    """Creates a follower relationship. Self-following is strictly blocked."""
    stmt = select(User).where(User.username == username.lower())
    target = (await db.execute(stmt)).scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if target.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot follow yourself.",
        )

    # Check if already following
    check_stmt = select(Follow).where(
        Follow.follower_id == current_user.id,
        Follow.following_id == target.id,
    )
    existing = (await db.execute(check_stmt)).scalar_one_or_none()
    if not existing:
        follow = Follow(follower_id=current_user.id, following_id=target.id)
        db.add(follow)
        await db.commit()

    return FollowResponse(username=target.username, following=True)


@router.delete(
    "/{username}/follow",
    response_model=FollowResponse,
    summary="Unfollow a user",
)
async def unfollow_user(
    username: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FollowResponse:
    """Removes a follower relationship."""
    stmt = select(User).where(User.username == username.lower())
    target = (await db.execute(stmt)).scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    del_stmt = delete(Follow).where(
        Follow.follower_id == current_user.id,
        Follow.following_id == target.id,
    )
    await db.execute(del_stmt)
    await db.commit()

    return FollowResponse(username=target.username, following=False)
