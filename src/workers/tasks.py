import asyncio
import os
import uuid
from decimal import Decimal
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from src.config import settings
from src.database import AsyncSessionLocal
from src.models.enums import (
    JobStatus,
    ModerationStatus,
    SceneGenerationStatus,
    StoryStatus,
    VideoVisibility,
)
from src.models.generation_job import GenerationJob
from src.models.scene import Scene
from src.models.story import Story
from src.models.video import Video
from src.services.llm.base import StoryContext
from src.services.llm.factory import get_llm_provider
from src.services.media.ffmpeg_runner import ffmpeg_runner
from src.services.media.storage import storage_service
from src.services.media.subtitles import generate_srt_subtitles
from src.services.tts.base import get_tts_provider
from src.services.tts.transliteration import transliterate_thanglish_to_tamil
from src.services.video.factory import get_video_provider
from src.workers.celery_app import celery_app


def run_async(coro):
    """Helper to run async coroutines within synchronous Celery workers."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@celery_app.task(name="analyze_story")
def analyze_story(story_id_str: str) -> dict:
    if settings.APP_ENV == "test":
        return {"status": "mock_dispatched", "story_id": story_id_str}
    return run_async(_async_analyze_story(uuid.UUID(story_id_str)))


async def _async_analyze_story(story_id: uuid.UUID) -> dict:
    async with AsyncSessionLocal() as session:
        stmt = select(Story).where(Story.id == story_id)
        story = (await session.execute(stmt)).scalar_one_or_none()
        if not story:
            return {"error": "Story not found"}

        story.status = StoryStatus.SCENE_PLANNING
        await session.commit()

        try:
            llm = get_llm_provider()
            ctx = StoryContext(
                story_id=story.id,
                title=story.title,
                original_story=story.original_story,
                input_language=story.input_language.value,
                selected_style=story.selected_style.value,
                requested_duration=story.requested_duration,
            )
            plan = await llm.generate_scenes(ctx)

            # Floating point duration normalization to ensure exact sum equals requested_duration
            durations = [Decimal(str(round(s.duration, 2))) for s in plan.scenes]
            total_dur = sum(durations)
            diff = Decimal(str(story.requested_duration)) - total_dur
            if diff != Decimal("0.00") and durations:
                durations[-1] += diff

            # Delete existing scenes if re-planning
            existing_scenes = await session.execute(select(Scene).where(Scene.story_id == story.id))
            for es in existing_scenes.scalars().all():
                await session.delete(es)

            created_scenes = []
            for idx, scene_data in enumerate(plan.scenes):
                scene_record = Scene(
                    story_id=story.id,
                    scene_number=idx + 1,
                    duration_seconds=durations[idx],
                    visual_prompt=scene_data.visual_prompt,
                    narration_text=scene_data.narration_text,
                    subtitle_text=scene_data.subtitle_text,
                    generation_status=SceneGenerationStatus.PENDING,
                )
                session.add(scene_record)
                created_scenes.append(scene_record)

            story.status = StoryStatus.GENERATING_SCENES
            await session.commit()

            # Refresh to get scene IDs and dispatch generation
            for sc in created_scenes:
                await session.refresh(sc)
                generate_scene_clip.delay(str(sc.id))

            return {
                "status": "success",
                "story_id": str(story.id),
                "scenes_count": len(created_scenes),
            }

        except Exception as e:
            story.status = StoryStatus.FAILED
            await session.commit()
            return {"error": str(e)}


@celery_app.task(name="generate_scene_clip")
def generate_scene_clip(scene_id_str: str) -> dict:
    if settings.APP_ENV == "test":
        return {"status": "mock_dispatched", "scene_id": scene_id_str}
    return run_async(_async_generate_scene_clip(uuid.UUID(scene_id_str)))


async def _async_generate_scene_clip(scene_id: uuid.UUID) -> dict:
    async with AsyncSessionLocal() as session:
        stmt = select(Scene).options(selectinload(Scene.story)).where(Scene.id == scene_id)
        scene = (await session.execute(stmt)).scalar_one_or_none()
        if not scene:
            return {"error": "Scene not found"}

        provider = get_video_provider()
        scene.generation_status = SceneGenerationStatus.PROCESSING
        await session.commit()

        try:
            res = provider.generate_clip(
                prompt=scene.visual_prompt,
                duration=float(scene.duration_seconds),
                style=scene.story.selected_style.value,
                aspect_ratio="16:9",
            )
            job_id = res.get("job_id")
            scene.provider_job_id = job_id

            # Track generation job
            idempotency_key = f"scene_{scene.id}_{job_id}"
            gen_job = GenerationJob(
                user_id=scene.story.user_id,
                story_id=scene.story.id,
                provider=provider.__class__.__name__,
                provider_job_id=job_id,
                status=JobStatus.PROCESSING,
                progress=10,
                idempotency_key=idempotency_key,
            )
            session.add(gen_job)
            await session.commit()

            return {"scene_id": str(scene.id), "provider_job_id": job_id, "status": "processing"}
        except Exception as e:
            scene.generation_status = SceneGenerationStatus.FAILED
            await session.commit()
            return {"error": str(e)}


@celery_app.task(name="assemble_final_video")
def assemble_final_video(story_id_str: str) -> dict:
    if settings.APP_ENV == "test":
        return {"status": "mock_dispatched", "story_id": story_id_str}
    return run_async(_async_assemble_final_video(uuid.UUID(story_id_str)))


async def _async_assemble_final_video(story_id: uuid.UUID) -> dict:
    async with AsyncSessionLocal() as session:
        stmt = (
            select(Story)
            .options(selectinload(Story.scenes))
            .where(Story.id == story_id)
        )
        story = (await session.execute(stmt)).scalar_one_or_none()
        if not story:
            return {"error": "Story not found"}

        story.status = StoryStatus.ASSEMBLING_VIDEO
        await session.commit()

        try:
            scenes = sorted(story.scenes, key=lambda s: s.scene_number)
            clip_paths = []
            for sc in scenes:
                asset_url = sc.generated_asset_url
                if not asset_url or not os.path.exists(asset_url):
                    # Ensure mock asset exists
                    sample_path = storage_service.get_clip_path(f"clip_{sc.id}.mp4")
                    if not os.path.exists(sample_path):
                        with open(sample_path, "wb") as f:
                            f.write(b"SAMPLE_VIDEO_DATA\n")
                    asset_url = sample_path
                    sc.generated_asset_url = sample_path
                clip_paths.append(asset_url)

            # Narration text handling: if Thanglish, transliterate to Tamil script for TTS
            narration_chunks = [s.narration_text for s in scenes if s.narration_text]
            full_narration = " ".join(narration_chunks)

            tts_input = full_narration
            tts_lang = "en"
            if story.input_language.value == "thanglish":
                tts_input = transliterate_thanglish_to_tamil(full_narration)
                tts_lang = "ta"
            elif story.input_language.value == "ta":
                tts_lang = "ta"

            tts = get_tts_provider()
            narration_audio = tts.synthesize(tts_input, tts_lang)

            # Subtitles: Romanized Thanglish is preserved in SRT
            srt_path = storage_service.get_subtitles_path(f"subs_{story.id}.srt")
            generate_srt_subtitles(scenes, srt_path)

            # Assembled output
            output_mp4 = storage_service.get_video_path(f"video_{story.id}.mp4")
            ffmpeg_runner.assemble(
                clip_paths=clip_paths,
                narration_audio_path=narration_audio,
                bgm_audio_path=None,
                srt_subtitles_path=srt_path,
                output_video_path=output_mp4,
            )

            # Total duration
            total_duration = sum([s.duration_seconds for s in scenes])

            # Upsert Video record
            existing_video = (
                await session.execute(select(Video).where(Video.story_id == story.id))
            ).scalar_one_or_none()

            if existing_video:
                existing_video.video_url = output_mp4
                existing_video.duration_seconds = total_duration
                existing_video.moderation_status = ModerationStatus.APPROVED
            else:
                video = Video(
                    user_id=story.user_id,
                    story_id=story.id,
                    title=story.title,
                    description=story.original_story[:250],
                    video_url=output_mp4,
                    duration_seconds=total_duration,
                    visual_style=story.selected_style.value,
                    ai_generated=True,
                    moderation_status=ModerationStatus.APPROVED,
                    visibility=VideoVisibility.PRIVATE,
                )
                session.add(video)

            story.status = StoryStatus.READY_FOR_PREVIEW
            await session.commit()

            return {"status": "success", "story_id": str(story.id), "video_path": output_mp4}

        except Exception as e:
            story.status = StoryStatus.FAILED
            await session.commit()
            return {"error": str(e)}
