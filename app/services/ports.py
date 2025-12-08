# ABOUTME: Port interfaces (abstractions) for infrastructure dependencies
# ABOUTME: Service layer depends on these; infrastructure implements them

from abc import ABC, abstractmethod
from uuid import UUID

import numpy as np

from app.domain.user import User, UserCreate
from app.domain.transfer import TransferJob


class UserRepository(ABC):
    """Port for user persistence."""

    @abstractmethod
    def get_by_id(self, user_id: UUID) -> User | None:
        pass

    @abstractmethod
    def get_by_email(self, email: str) -> User | None:
        pass

    @abstractmethod
    def create(self, user: UserCreate, hashed_password: str) -> User:
        pass

    @abstractmethod
    def update(self, user: User) -> User:
        pass


class TransferJobRepository(ABC):
    """Port for transfer job persistence."""

    @abstractmethod
    def get_by_id(self, job_id: UUID) -> TransferJob | None:
        pass

    @abstractmethod
    def get_by_user(self, user_id: UUID, limit: int = 10) -> list[TransferJob]:
        pass

    @abstractmethod
    def create(self, job: TransferJob) -> TransferJob:
        pass

    @abstractmethod
    def update(self, job: TransferJob) -> TransferJob:
        pass


class ImageStorage(ABC):
    """Port for image file storage."""

    @abstractmethod
    def save(self, image: np.ndarray, path: str) -> str:
        """Save image and return storage path."""
        pass

    @abstractmethod
    def load(self, path: str) -> np.ndarray:
        """Load image from storage path."""
        pass

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete image from storage."""
        pass

    @abstractmethod
    def get_url(self, path: str) -> str:
        """Get public URL for image."""
        pass


class PasswordHasher(ABC):
    """Port for password hashing."""

    @abstractmethod
    def hash(self, password: str) -> str:
        pass

    @abstractmethod
    def verify(self, password: str, hashed: str) -> bool:
        pass
