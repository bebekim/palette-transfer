# ABOUTME: User service - orchestrates user-related workflows
# ABOUTME: Handles user registration, authentication, profile updates

from datetime import datetime
from uuid import UUID

from app.domain.user import User, UserCreate
from app.domain.exceptions import UserNotFoundError, ValidationError
from app.services.ports import UserRepository, PasswordHasher


class UserService:
    """Orchestrates user operations."""

    def __init__(
        self,
        user_repository: UserRepository,
        password_hasher: PasswordHasher,
    ):
        self.user_repository = user_repository
        self.password_hasher = password_hasher

    def register(self, data: UserCreate) -> User:
        """Register a new user.

        Args:
            data: User creation data with email and password

        Returns:
            Created user entity

        Raises:
            ValidationError: If email already exists
        """
        existing = self.user_repository.get_by_email(data.email)
        if existing:
            raise ValidationError("Email already registered", field="email")

        hashed_password = self.password_hasher.hash(data.password)
        return self.user_repository.create(data, hashed_password)

    def authenticate(self, email: str, password: str) -> User | None:
        """Authenticate user by email and password.

        Args:
            email: User email
            password: Plain text password

        Returns:
            User if authentication successful, None otherwise
        """
        user = self.user_repository.get_by_email(email.lower().strip())
        if not user:
            return None

        # Note: We need to get hashed password from repository
        # This is a simplification - in practice, repository would
        # return a model with hashed_password accessible
        if not self.password_hasher.verify(password, user.hashed_password):
            return None

        return user

    def get_user(self, user_id: UUID) -> User:
        """Get user by ID.

        Args:
            user_id: User UUID

        Returns:
            User entity

        Raises:
            UserNotFoundError: If user not found
        """
        user = self.user_repository.get_by_id(user_id)
        if not user:
            raise UserNotFoundError(str(user_id))
        return user

    def update_last_login(self, user_id: UUID) -> User:
        """Update user's last login timestamp.

        Args:
            user_id: User UUID

        Returns:
            Updated user entity
        """
        user = self.get_user(user_id)
        user.last_login = datetime.utcnow()
        user.updated_at = datetime.utcnow()
        return self.user_repository.update(user)
