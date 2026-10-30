import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from src.api.deps import get_current_user
from src.database import get_db
from src.models.enums import SceneGenerationStatus
from src.models.scene import Scene
from src.models.user import User
from src.schemas.story import SceneStatusResponse
from src.services.video.factory import get_video_provider
from src.workers.tasks import generate_scene_clip

router = APIRouter(prefix="/scenes", tags=["Scenes"])


@router.post(
    "/{id}/generate",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Dispatch video clip generation for a scene",
)
async def trigger_scene_generation(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Manually dispatches clip generation task for a scene."""
    stmt = select(Scene).options(selectinload(Scene.story)).where(Scene.id == id)
    scene = (await db.execute(stmt)).scalar_one_or_none()
    if not scene:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found")
    if scene.story.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    generate_scene_clip.delay(str(scene.id))
    return {"message": "Clip generation queued", "scene_id": str(scene.id)}


@router.get(
    "/{id}/status",
    response_model=SceneStatusResponse,
    summary="Poll scene clip generation status",
)
async def get_scene_status(
    id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SceneStatusResponse:
    """
    Polls the provider status for a scene clip.
    If provider returns completed, updates generated_asset_url and sets status to 'completed'.
    """
    stmt = select(Scene).options(selectinload(Scene.story)).where(Scene.id == id)
    scene = (await db.execute(stmt)).scalar_one_or_none()
    if not scene:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found")
    if scene.story.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    if scene.generation_status == SceneGenerationStatus.PROCESSING and scene.provider_job_id:
        provider = get_video_provider()
        res = provider.get_status(scene.provider_job_id)
        if res.get("status") == "completed":
            scene.generation_status = SceneGenerationStatus.COMPLETED
            scene.generated_asset_url = res.get("video_url")
            await db.commit()
        elif res.get("status") == "failed":
            scene.generation_status = SceneGenerationStatus.FAILED
            await db.commit()

    return SceneStatusResponse(
        scene_id=scene.id,
        story_id=scene.story_id,
        scene_number=scene.scene_number,
        generation_status=scene.generation_status,
        provider_job_id=scene.provider_job_id,
        generated_asset_url=scene.generated_asset_url,
        duration_seconds=scene.duration_seconds,
        created_at=scene.created_at,
        updated_at=scene.updated_at,
    )
