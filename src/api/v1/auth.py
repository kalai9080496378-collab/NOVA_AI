import time
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.deps import get_current_user, oauth2_scheme
from src.core.rate_limiter import (
    check_login_rate_limit,
    record_failed_login_attempt,
    reset_login_rate_limit,
    revoke_token,
)
from src.core.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)
from src.database import get_db
from src.models.user import User
from src.schemas.auth import LoginRequest, SignupRequest, Token
from src.schemas.user import UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/signup",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def signup(
    payload: SignupRequest,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Creates a new user account with hashed password and initial generation quota."""
    # Check duplicate username or email
    stmt = select(User).where(
        or_(
            User.username == payload.username.lower(),
            User.email == payload.email.lower(),
        )
    )
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        if existing.username == payload.username.lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username is already taken.",
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        username=payload.username.lower(),
        email=payload.email.lower(),
        password_hash=get_password_hash(payload.password),
        display_name=payload.display_name or payload.username,
        monthly_generation_quota=3,
        generation_quota_remaining=3,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post(
    "/login",
    response_model=Token,
    summary="Authenticate and receive JWT access token",
)
async def login(
    request: Request,
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Authenticates credentials with rate limiting protection and issues JWT access token."""
    client_ip = request.client.host if request.client else "unknown"
    check_login_rate_limit(client_ip, payload.email_or_username)

    stmt = select(User).where(
        or_(
            User.email == payload.email_or_username.lower(),
            User.username == payload.email_or_username.lower(),
        )
    )
    user = (await db.execute(stmt)).scalar_one_or_none()

    if not user or not verify_password(payload.password, user.password_hash):
        record_failed_login_attempt(client_ip, payload.email_or_username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email/username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Reset rate limiter upon successful login
    reset_login_rate_limit(client_ip, payload.email_or_username)

    token_str, _, expires_in = create_access_token(user.id)
    return Token(
        access_token=token_str,
        token_type="bearer",
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    summary="Revoke the current JWT access token",
)
async def logout(
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Revokes the current access token server-side via unique token ID (jti) blocklist."""
    payload = decode_access_token(token)
    jti = payload.get("jti")
    exp = payload.get("exp")
    if jti and exp:
        remaining_ttl = max(int(exp - time.time()), 1)
        revoke_token(jti, remaining_ttl)
    return {"message": "Successfully logged out. Token has been revoked."}
