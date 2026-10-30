import uuid
from fastapi import HTTPException, status
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from src.models.user import User


async def check_and_deduct_user_quota(db: AsyncSession, user_id: uuid.UUID) -> None:
    """
    Checks user's monthly generation quota and atomically deducts 1 credit at submission time.
    Raises HTTPException(403) if quota is exhausted.
    """
    # Atomic deduction query: only decrements if generation_quota_remaining > 0
    stmt = (
        update(User)
        .where(User.id == user_id, User.generation_quota_remaining > 0)
        .values(generation_quota_remaining=User.generation_quota_remaining - 1)
    )
    result = await db.execute(stmt)
    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Monthly video generation quota exceeded. Free tier limit is 3 videos per month.",
        )
    await db.commit()


async def refund_user_quota(db: AsyncSession, user_id: uuid.UUID) -> None:
    """Refunds 1 generation quota credit in the event of pre-provider unrecoverable failures."""
    stmt = (
        update(User)
        .where(User.id == user_id)
        .values(generation_quota_remaining=User.generation_quota_remaining + 1)
    )
    await db.execute(stmt)
    await db.commit()
