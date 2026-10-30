from pydantic import EmailStr, Field
from src.schemas.common import BaseSchema


class SignupRequest(BaseSchema):
    email: EmailStr
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[A-Za-z0-9_]+$",
        description="Username must be 3+ chars, alphanumeric and underscore only",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password must be at least 8 characters long",
    )
    display_name: str | None = Field(None, max_length=100)


class LoginRequest(BaseSchema):
    email_or_username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=1)


class Token(BaseSchema):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseSchema):
    sub: str
    jti: str
    exp: int
    type: str = "access"
