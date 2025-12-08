# ABOUTME: Domain layer - pure business entities and rules
# ABOUTME: No framework dependencies (Flask, SQLAlchemy) allowed here

from app.domain.user import User, UserCreate
from app.domain.transfer import (
    TransferMethod,
    TransferParams,
    TransferJob,
    TransferResult,
    SkinDetectionParams,
    SkinDetectionResult,
)
from app.domain.exceptions import (
    DomainError,
    ValidationError,
    ImageProcessingError,
    UserNotFoundError,
)

__all__ = [
    "User",
    "UserCreate",
    "TransferMethod",
    "TransferParams",
    "TransferJob",
    "TransferResult",
    "SkinDetectionParams",
    "SkinDetectionResult",
    "DomainError",
    "ValidationError",
    "ImageProcessingError",
    "UserNotFoundError",
]
