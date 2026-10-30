# NOVA AI — System Implementation Plan (Full-Stack)

## 1. Overview
NOVA AI is an AI-powered story-to-video generation platform and creator social ecosystem. This implementation plan covers the complete full-stack architecture across ten cohesive milestones:
1. **Backend Foundation**: FastAPI + SQLAlchemy 2.0 (PostgreSQL) + Alembic migrations + Celery (Redis).
2. **Authentication & User Management**: Bcrypt password hashing (`passlib`), JWT access tokens, token revocation blocklist, login rate limiting, and profile management (`/users/me`, `/users/{username}`).
3. **Cost Protection & Monthly Generation Quotas**: Strict quota tracking on `users` (`monthly_generation_quota`, `generation_quota_remaining`), with atomic deduction at job submission time (3 videos/month default for free tier) to prevent unauthorized cloud spend.
4. **Explicit Thanglish Input & Transliteration Pipeline**: Explicit user language selection (no auto-detection), dedicated LLM preservation prompt, phonetic conversion to Tamil script for Tamil TTS, and Romanized captions preservation avoiding FFmpeg complex text shaping (`libass`/HarfBuzz) bugs.
5. **Content Moderation & AI-Only Publishing Enforcement**:
   - Ingestion scanner: Keyword blocklist checks `original_story` before story creation.
   - Pipeline moderation gate: Video generation sets `moderation_status = "pending"`. Only `"approved"` videos can be set to `visibility = "public"`.
   - Admin moderation endpoints for approving/rejecting videos.
   - Strict AI-only verification: Blocks manual video uploads; verifies video originated from a valid story generation job with completed generation status before publishing.
   - Community reporting system: `reports` table and endpoints (`POST /reports`, `GET /admin/reports`, `PATCH /admin/reports/{id}`).
6. **Story Submission & LLM Scene Planning**: Story creation (`POST /stories`), async planning dispatch (`POST /stories/{id}/plan`), modular `LLMProvider` interface (Gemini/Mock), structured JSON prompt engineering, and Celery worker scene persistence (`analyze_story`).
7. **Real Video Generation Provider: RunAPI Veo 3.1 (`runapi-veo-3-1`)**: High-fidelity video generation using Veo 3.1 Fast, 2-stage retry logic for rate limits/timeouts, and an immediate asset persistence pipeline downloading clips before RunAPI's 7-day expiration window.
8. **Video Assembly Pipeline using FFmpeg & TTS**: Subtitle generation (SRT timecode mapping), TTS narration synthesis (`TTSProvider`), FFmpeg concat demuxer (`-f concat -safe 0`), audio ducking & mixing (`amix` with background music at 15%), subtitle burning (`subtitles` filter), and Celery task `assemble_final_video(story_id, scene_ids)` creating the final `Video` record.
9. **Social Features & Discovery Feed**: Social graph (`likes`, `comments`, `follows`, `saved_videos`), atomic counter denormalization (`like_count`, `comment_count`), strict self-action prevention constraints, cursor-based feed pagination (`GET /feed`), and saved bookmarks.
10. **Next.js Frontend Studio**: Next.js App Router client styled with Tailwind CSS in dark obsidian (`#09090D`) with rose gold accents (`#C58B83`), featuring `/studio` (story input form with Thanglish notice), `/studio/{storyId}` (3s polling progress monitor through pipeline stages), custom preview player, and the one-click public publish flow (`PATCH /videos/{id}`).

---

## 2. Content Moderation & AI-Only Enforcement Architecture

### 2.1. Ingestion Keyword Blocklist
- Before creating a `Story` record in `POST /api/v1/stories`, `src/core/moderation.py` scans `original_story` and `title`.
- Checks for prohibited keywords across categories: illegal weapons, hate speech, severe violence, exploitation, self-harm, and harassment.
- If a match is detected: raises `HTTPException(status_code=400, detail="Story content violated safety guidelines and cannot be submitted.")`.

### 2.2. Post-Generation Moderation Gate
- When `assemble_final_video` completes, the newly assembled `Video` record is inserted with:
  - `moderation_status = ModerationStatus.PENDING`
  - `visibility = VideoVisibility.PRIVATE`
- A video **cannot** be published (`visibility="public"`) while `moderation_status == "pending"` or `"rejected"`.
- **Admin Review Endpoints (`src/api/v1/admin.py`)**:
  - `GET /api/v1/admin/moderation/videos`: Lists pending videos with story context and video URL.
  - `POST /api/v1/admin/moderation/videos/{id}/review`: Accepts `{ "action": "approve" | "reject", "reason": str | None }`.
    - If `approve`: sets `moderation_status = ModerationStatus.APPROVED`.
    - If `reject`: sets `moderation_status = ModerationStatus.REJECTED`, forces `visibility = VideoVisibility.PRIVATE`.
  - Protected by `get_current_admin_user` dependency (requires `user.is_admin == True`).

### 2.3. Strict AI-Only Publishing Verification
To guarantee that the platform hosts solely AI-generated content produced on NOVA AI, `PATCH /api/v1/videos/{id}` verifies:
1. `video.ai_generated is True`.
2. `video.story_id is not None` (must be tied to a database-tracked story).
3. Associated `Story` and `GenerationJob` exist, verifying genuine pipeline origin.
4. `GenerationJob.status` is verified completed (`succeeded` / `ready_for_preview`).
5. `video.moderation_status == ModerationStatus.APPROVED`. If still pending, returns `HTTPException(400, "Video is awaiting content moderation approval before public publishing.")`.
6. Arbitrary video uploads without an active story generation job are strictly rejected.

### 2.4. Reporting System (`reports` Table & Endpoints)
- **Table Schema**:
  - `id`: UUID (PK)
  - `reporter_id`: UUID FK -> `users.id` (CASCADE)
  - `target_type`: Enum (`video`, `comment`, `user`)
  - `target_id`: UUID (target entity ID)
  - `reason`: `Text` (`CHECK (length(trim(reason)) > 0 AND length(reason) <= 500)`)
  - `status`: Enum (`ReportStatus`: `pending`, `reviewed`, `dismissed`, `action_taken`), default `pending`
  - `reviewed_by`: UUID FK -> `users.id` (nullable)
  - `reviewed_at`: `DateTime(timezone=True)`, nullable
  - `created_at`, `updated_at`: timestamps
- **Endpoints**:
  - `POST /api/v1/reports`:
    - Auth required. Accepts `{ target_type, target_id, reason }`.
    - Validates target entity exists in DB.
    - Creates report with `status = "pending"`.
  - `GET /api/v1/admin/reports`:
    - Admin only. Lists reports filtered by `status` and `target_type` with pagination.
  - `PATCH /api/v1/admin/reports/{id}`:
    - Admin only. Accepts `{ "status": "reviewed" | "dismissed" | "action_taken", "action_notes": str }`.
    - Updates `reviewed_by = current_admin.id` and `reviewed_at = func.now()`.

---

## 3. Database Schema & Models

```
NOVA_AI/
├── src/models/
│   ├── base.py
│   ├── enums.py            # ModerationStatus, ReportStatus, TargetType, etc.
│   ├── user.py             # Added is_admin: Boolean (default False), quota fields
│   ├── story.py
│   ├── scene.py
│   ├── generation_job.py
│   ├── video.py            # moderation_status, ai_generated, story_id FK (UNIQUE)
│   ├── like.py
│   ├── comment.py
│   ├── follow.py
│   ├── saved_video.py
│   └── report.py           # reports table (reporter_id, target_type, target_id, status)
```

- **`users`**:
  - Added `is_admin`: `Boolean`, default `False`, `nullable=False`.
- **`videos`**:
  - `moderation_status`: Enum (`pending`, `approved`, `rejected`, `flagged`), default `pending`.
  - `ai_generated`: `Boolean`, default `True`.
  - `story_id`: UUID FK -> `stories.id`, `UNIQUE`.
- **`reports`**:
  - Full reporting entity tracking community flags.

---

## 4. Verification & Automated Test Strategy

1. **Moderation Blocklist Tests (`tests/test_moderation.py`)**:
   - `test_story_submission_blocked_by_prohibited_keywords`: Story containing blocked terms is rejected with 400 Bad Request.
   - `test_clean_story_submission_accepted`: Valid story proceeds without issue.
2. **AI-Only Publishing Gate Tests**:
   - `test_publish_blocked_when_moderation_pending`: Video with `moderation_status="pending"` cannot be made public.
   - `test_publish_blocked_when_moderation_rejected`: Video with `moderation_status="rejected"` returns 403 Forbidden.
   - `test_publish_blocked_without_story_id`: Video lacking genuine story lineage is rejected.
   - `test_publish_succeeds_when_approved`: Video with `moderation_status="approved"` successfully publishes to feed.
3. **Admin Moderation Endpoints Tests**:
   - `test_non_admin_cannot_review_videos`: Non-admin receives 403 Forbidden on `/admin/*` routes.
   - `test_admin_approves_video`: Admin transitions video to `approved`, enabling publishing.
4. **Reporting System Tests**:
   - `test_user_can_report_video`: Valid report created with status `pending`.
   - `test_admin_can_resolve_report`: Admin marks report as `reviewed` / `action_taken`.

---

## 5. Step-by-Step Implementation Phasing

- **Phase 1: Dependencies & Configuration**
  - Python dependencies, `.env.example`, `src/config.py`.
- **Phase 2: Database Models & Alembic Migration**
  - Add `is_admin` to `users`, create `reports` model, update `videos` moderation defaults.
  - Generate initial Alembic migration `0001_initial_schema.py`.
- **Phase 3: Security, Auth & Admin Dependencies**
  - Bcrypt, PyJWT, `get_current_user`, `get_current_admin_user`, quota deductions.
- **Phase 4: Content Moderation & AI-Only Publishing Checks**
  - Ingestion blocklist in `src/core/moderation.py`.
  - Publish verification logic in `PATCH /videos/{id}`.
  - Admin review endpoints in `src/api/v1/admin.py`.
  - Reports endpoints in `src/api/v1/reports.py`.
- **Phase 5: Story Submission, Thanglish & Scene Planning**
  - Thanglish preservation and style prompts, Celery `analyze_story` task.
- **Phase 6: RunAPI Veo 3.1 & Asset Ingestion**
  - `RunAPIVeoProvider` with 2 retries and immediate asset download.
- **Phase 7: Video Assembly & Subtitles**
  - Thanglish TTS transliteration, FFmpeg concat, 15% audio ducking, subtitle burn-in.
- **Phase 8: Social Features & Discovery Feed**
  - Likes (no self-like), comments, follows, bookmarks, and cursor pagination `/feed`.
- **Phase 9: Next.js Frontend Studio**
  - `/studio` with Thanglish notice, `/studio/[storyId]` progress page with preview and publish flow.
- **Phase 10: End-to-End Verification**
  - Complete automated test suite covering moderation, AI-only gate, reports, and assembly.
