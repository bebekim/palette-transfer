# ABOUTME: User domain entity - pure business representation of a user
# ABOUTME: No ORM or framework dependencies

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Data required to create a new user."""

    email: EmailStr
    password: str = Field(min_length=8)


class User(BaseModel):
    """User domain entity."""

    id_: UUID = Field(default_factory=uuid4)
    email: EmailStr
    username: str | None = None
    display_name: str | None = None
    avatar_url: str | None = None

    is_active: bool = True
    is_verified: bool = False

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: datetime | None = None

    class Config:
        from_attributes = True
