# ABOUTME: Dependency injection factories for API routes
# ABOUTME: Creates service instances with concrete infrastructure adapters

from functools import lru_cache

from app.services.transfer_service import TransferService
from app.services.user_service import UserService
from app.infrastructure.database.repositories import (
    SQLUserRepository,
    SQLTransferJobRepository,
)
from app.infrastructure.storage.local import LocalImageStorage
from app.infrastructure.auth import WerkzeugPasswordHasher


@lru_cache
def get_transfer_service() -> TransferService:
    """Factory for TransferService with injected dependencies."""
    return TransferService(
        job_repository=SQLTransferJobRepository(),
        image_storage=LocalImageStorage(base_path="uploads"),
    )


@lru_cache
def get_user_service() -> UserService:
    """Factory for UserService with injected dependencies."""
    return UserService(
        user_repository=SQLUserRepository(),
        password_hasher=WerkzeugPasswordHasher(),
    )


def get_transfer_service_stateless() -> TransferService:
    """Stateless TransferService without persistence (for quick operations)."""
    return TransferService()
