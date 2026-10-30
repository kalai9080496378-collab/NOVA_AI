from fastapi import APIRouter
from src.api.v1.admin import router as admin_router
from src.api.v1.auth import router as auth_router
from src.api.v1.feed import router as feed_router
from src.api.v1.reports import router as reports_router
from src.api.v1.scenes import router as scenes_router
from src.api.v1.stories import router as stories_router
from src.api.v1.users import router as users_router
from src.api.v1.videos import router as videos_router

api_router = APIRouter()

api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(stories_router)
api_router.include_router(scenes_router)
api_router.include_router(videos_router)
api_router.include_router(feed_router)
api_router.include_router(reports_router)
api_router.include_router(admin_router)
