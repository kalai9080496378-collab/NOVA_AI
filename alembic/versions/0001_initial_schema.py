"""Initial schema for NOVA AI

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-10-30 11:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. users table
    op.create_table(
        "users",
        sa.Column("id", sa.CHAR(36), primary_key=True, nullable=False),
        sa.Column("username", sa.String(50), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(100), nullable=True),
        sa.Column("bio", sa.Text(), nullable=True),
        sa.Column("avatar_url", sa.String(512), nullable=True),
        sa.Column("account_status", sa.String(50), nullable=False, server_default="active"),
        sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("monthly_generation_quota", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("generation_quota_remaining", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("quota_reset_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("length(username) >= 3 AND username ~ '^[A-Za-z0-9_]+$'", name="ck_users_username_format"),
        sa.CheckConstraint("monthly_generation_quota >= 0", name="ck_users_monthly_quota_non_negative"),
        sa.CheckConstraint("generation_quota_remaining >= 0", name="ck_users_quota_remaining_non_negative"),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # 2. stories table
    op.create_table(
        "stories",
        sa.Column("id", sa.CHAR(36), primary_key=True, nullable=False),
        sa.Column("user_id", sa.CHAR(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("original_story", sa.Text(), nullable=False),
        sa.Column("input_language", sa.String(20), nullable=False),
        sa.Column("selected_style", sa.String(50), nullable=False),
        sa.Column("requested_duration", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(50), nullable=False, server_default="STORY_SUBMITTED"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("requested_duration IN (10, 15, 20, 30)", name="ck_stories_requested_duration"),
    )
    op.create_index("ix_stories_user_id", "stories", ["user_id"])
    op.create_index("ix_stories_status", "stories", ["status"])

    # 3. scenes table
    op.create_table(
        "scenes",
        sa.Column("id", sa.CHAR(36), primary_key=True, nullable=False),
        sa.Column("story_id", sa.CHAR(36), sa.ForeignKey("stories.id", ondelete="CASCADE"), nullable=False),
        sa.Column("scene_number", sa.Integer(), nullable=False),
        sa.Column("duration_seconds", sa.Numeric(5, 2), nullable=False),
        sa.Column("visual_prompt", sa.Text(), nullable=False),
        sa.Column("narration_text", sa.Text(), nullable=True),
        sa.Column("subtitle_text", sa.Text(), nullable=True),
        sa.Column("generation_status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("generated_asset_url", sa.String(1024), nullable=True),
        sa.Column("provider_job_id", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("story_id", "scene_number", name="uq_scenes_story_scene_number"),
    )
    op.create_index("ix_scenes_story_id", "scenes", ["story_id"])
    op.create_index("ix_scenes_generation_status", "scenes", ["generation_status"])
    op.create_index("ix_scenes_provider_job_id", "scenes", ["provider_job_id"])

    # 4. generation_jobs table
    op.create_table(
        "generation_jobs",
        sa.Column("id", sa.CHAR(36), primary_key=True, nullable=False),
        sa.Column("user_id", sa.CHAR(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("story_id", sa.CHAR(36), sa.ForeignKey("stories.id", ondelete="CASCADE"), nullable=True),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("provider_job_id", sa.String(255), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_code", sa.String(100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("idempotency_key", name="uq_generation_jobs_idempotency_key"),
        sa.CheckConstraint("progress >= 0 AND progress <= 100", name="ck_generation_jobs_progress_range"),
    )
    op.create_index("ix_generation_jobs_user_id", "generation_jobs", ["user_id"])
    op.create_index("ix_generation_jobs_story_id", "generation_jobs", ["story_id"])
    op.create_index("ix_generation_jobs_status", "generation_jobs", ["status"])

    # 5. videos table
    op.create_table(
        "videos",
        sa.Column("id", sa.CHAR(36), primary_key=True, nullable=False),
        sa.Column("user_id", sa.CHAR(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("story_id", sa.CHAR(36), sa.ForeignKey("stories.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("video_url", sa.String(1024), nullable=False),
        sa.Column("thumbnail_url", sa.String(1024), nullable=True),
        sa.Column("duration_seconds", sa.Numeric(6, 2), nullable=False),
        sa.Column("visual_style", sa.String(50), nullable=False),
        sa.Column("ai_generated", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("moderation_status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("visibility", sa.String(50), nullable=False, server_default="private"),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("like_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("comment_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("story_id", name="uq_videos_story_id"),
        sa.CheckConstraint("like_count >= 0", name="ck_videos_like_count_non_negative"),
        sa.CheckConstraint("comment_count >= 0", name="ck_videos_comment_count_non_negative"),
    )
    op.create_index("ix_videos_user_id", "videos", ["user_id"])
    op.create_index("ix_videos_visibility", "videos", ["visibility"])
    op.create_index("ix_videos_moderation_status", "videos", ["moderation_status"])
    op.create_index("ix_videos_published_at", "videos", ["published_at"])

    # 6. likes table
    op.create_table(
        "likes",
        sa.Column("id", sa.CHAR(36), primary_key=True, nullable=False),
        sa.Column("user_id", sa.CHAR(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("video_id", sa.CHAR(36), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("user_id", "video_id", name="uq_likes_user_video"),
    )
    op.create_index("ix_likes_user_id", "likes", ["user_id"])
    op.create_index("ix_likes_video_id", "likes", ["video_id"])

    # 7. comments table
    op.create_table(
        "comments",
        sa.Column("id", sa.CHAR(36), primary_key=True, nullable=False),
        sa.Column("user_id", sa.CHAR(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("video_id", sa.CHAR(36), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_comment_id", sa.CHAR(36), sa.ForeignKey("comments.id", ondelete="CASCADE"), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("length(trim(content)) > 0 AND length(content) <= 500", name="ck_comments_content_length"),
    )
    op.create_index("ix_comments_user_id", "comments", ["user_id"])
    op.create_index("ix_comments_video_id", "comments", ["video_id"])
    op.create_index("ix_comments_parent_comment_id", "comments", ["parent_comment_id"])

    # 8. follows table
    op.create_table(
        "follows",
        sa.Column("id", sa.CHAR(36), primary_key=True, nullable=False),
        sa.Column("follower_id", sa.CHAR(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("following_id", sa.CHAR(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("follower_id", "following_id", name="uq_follows_follower_following"),
        sa.CheckConstraint("follower_id != following_id", name="ck_follows_not_self"),
    )
    op.create_index("ix_follows_follower_id", "follows", ["follower_id"])
    op.create_index("ix_follows_following_id", "follows", ["following_id"])

    # 9. saved_videos table
    op.create_table(
        "saved_videos",
        sa.Column("id", sa.CHAR(36), primary_key=True, nullable=False),
        sa.Column("user_id", sa.CHAR(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("video_id", sa.CHAR(36), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("user_id", "video_id", name="uq_saved_videos_user_video"),
    )
    op.create_index("ix_saved_videos_user_id", "saved_videos", ["user_id"])
    op.create_index("ix_saved_videos_video_id", "saved_videos", ["video_id"])

    # 10. reports table
    op.create_table(
        "reports",
        sa.Column("id", sa.CHAR(36), primary_key=True, nullable=False),
        sa.Column("reporter_id", sa.CHAR(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_type", sa.String(20), nullable=False),
        sa.Column("target_id", sa.CHAR(36), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("reviewed_by", sa.CHAR(36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("length(trim(reason)) > 0 AND length(reason) <= 500", name="ck_reports_reason_length"),
    )
    op.create_index("ix_reports_reporter_id", "reports", ["reporter_id"])
    op.create_index("ix_reports_target_type", "reports", ["target_type"])
    op.create_index("ix_reports_target_id", "reports", ["target_id"])
    op.create_index("ix_reports_status", "reports", ["status"])


def downgrade() -> None:
    op.drop_table("reports")
    op.drop_table("saved_videos")
    op.drop_table("follows")
    op.drop_table("comments")
    op.drop_table("likes")
    op.drop_table("videos")
    op.drop_table("generation_jobs")
    op.drop_table("scenes")
    op.drop_table("stories")
    op.drop_table("users")
