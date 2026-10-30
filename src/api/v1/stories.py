import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.api.deps import get_current_user
from src.config import settings
from src.core.moderation import scan_story_content
from src.core.quota import check_and_deduct_user_quota
from src.database import get_db
from src.models.enums import SceneGenerationStatus, StoryStatus
from src.models.scene import Scene
from src.models.story import Story
from src.models.user import User
from src.models.video import Video
from src.schemas.story import (
    StoryCreate,
    StoryResponse,
    StoryStatusResponse,
)
from src.workers.tasks import analyze_story, assemble_final_video

router = APIRouter(prefix="/stories", tags=["Stories"])


@router.post(
    "",
    response_model=StoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a new story for video planning",
)
async def create_story(
    payload: StoryCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Story:
    """Scans content safety blocklist and creates a Story record with status STORY_SUBMITTED."""
    scan_story_content(payload.title, payload.original_story)

    story = Story(
        user_id=current_user.id,
        title=payload.title,
        original_story=payload.original_story,
        input_language=payload.input_language,
        selected_style=payload.selected_style,
        requested_duration=payload.requested_duration,
        status=StoryStatus.STORY_SUBMITTED,
    )
    db.add(story)
    await db.commit()
    await db.refresh(story)
    story.scenes = []
    return story


@router.post(
    "/{id}/plan",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger LLM scene planning and deduct generation quota",
)
async def plan_story_scenes(
    id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Verifies story ownership, deducts user quota credit, and triggers Celery analyze_story task."""
    stmt = select(Story).where(Story.id == id)
    story = (await db.execute(stmt)).scalar_one_or_none()
    if not story:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Story not found")

    if story.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to plan this story.",
        )

    # Cost protection: Deduct quota on job submission
    await check_and_deduct_user_quota(db, current_user.id)

    story.status = StoryStatus.STORY_ANALYSIS
    await db.commit()

    # Dispatch Celery background task if not test
    if settings.APP_ENV != "test":
        try:
            analyze_story.delay(str(story.id))
        except Exception:
            pass

    return {
        "message": "Scene planning initiated",
        "story_id": str(story.id),
        "status": "STORY_ANALYSIS",
    }


@router.get(
    "/{id}/status",
    response_model=StoryStatusResponse,
    summary="Poll story generation progress and stages",
)
async def get_story_status(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StoryStatusResponse:
    """
    Returns real-time execution status tailored for the frontend 3-second progress poller:
    STORY_ANALYSIS -> SCENE_PLANNING -> GENERATING_SCENES -> ASSEMBLING_VIDEO -> READY_FOR_PREVIEW
    """
    stmt = select(Story).options(selectinload(Story.scenes)).where(Story.id == id)
    story = (await db.execute(stmt)).scalar_one_or_none()
    if not story:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Story not found")

    if story.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    scenes = story.scenes or []
    completed_scenes = sum(1 for s in scenes if s.generation_status == SceneGenerationStatus.COMPLETED)
    total_scenes = len(scenes)

    # Check if all scenes completed and auto-trigger assembly if needed
    if (
        story.status == StoryStatus.GENERATING_SCENES
        and total_scenes > 0
        and completed_scenes == total_scenes
    ):
        assemble_final_video.delay(str(story.id))
        story.status = StoryStatus.ASSEMBLING_VIDEO
        await db.commit()

    # Progress mapping
    progress_map = {
        StoryStatus.STORY_SUBMITTED: 10,
        StoryStatus.STORY_ANALYSIS: 20,
        StoryStatus.SCENE_PLANNING: 35,
        StoryStatus.SCENES_GENERATED: 50,
        StoryStatus.GENERATING_SCENES: 50 + int((completed_scenes / max(total_scenes, 1)) * 25),
        StoryStatus.ASSEMBLING_VIDEO: 85,
        StoryStatus.READY_FOR_PREVIEW: 100,
        StoryStatus.COMPLETED: 100,
        StoryStatus.FAILED: 0,
    }
    progress_percent = progress_map.get(story.status, 20)

    # Fetch video if assembled
    video_stmt = select(Video).where(Video.story_id == story.id)
    video = (await db.execute(video_stmt)).scalar_one_or_none()

    stage_str = story.status.value if hasattr(story.status, "value") else str(story.status)

    return StoryStatusResponse(
        story_id=story.id,
        title=story.title,
        status=story.status,
        stage=stage_str,
        progress_percent=progress_percent,
        scenes_count=total_scenes,
        completed_scenes=completed_scenes,
        video_id=video.id if video else None,
        video_url=video.video_url if video else None,
        error=None,
    )


@router.get(
    "/{id}",
    response_model=StoryResponse,
    summary="Get story details with ordered scenes",
)
async def get_story(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Story:
    """Returns story record and all generated scenes."""
    stmt = select(Story).options(selectinload(Story.scenes)).where(Story.id == id)
    story = (await db.execute(stmt)).scalar_one_or_none()
    if not story:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Story not found")
    if story.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    return story
