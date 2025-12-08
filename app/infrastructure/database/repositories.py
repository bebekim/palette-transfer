# ABOUTME: SQLAlchemy repository implementations
# ABOUTME: Implements ports.UserRepository and ports.TransferJobRepository

from uuid import UUID

from app.extensions import db
from app.domain.user import User, UserCreate
from app.domain.transfer import TransferJob, TransferParams, TransferStatus
from app.services.ports import UserRepository, TransferJobRepository
from app.infrastructure.database.models import UserModel, TransferJobModel


class SQLUserRepository(UserRepository):
    """SQLAlchemy implementation of UserRepository port."""

    def get_by_id(self, user_id: UUID) -> User | None:
        model = UserModel.query.get(int(user_id))
        if not model:
            return None
        return self._to_domain(model)

    def get_by_email(self, email: str) -> User | None:
        model = UserModel.query.filter_by(email=email.lower().strip()).first()
        if not model:
            return None
        return self._to_domain(model)

    def create(self, user: UserCreate, hashed_password: str) -> User:
        model = UserModel(
            email=user.email.lower().strip(),
            password_hash=hashed_password,
        )
        db.session.add(model)
        db.session.commit()
        return self._to_domain(model)

    def update(self, user: User) -> User:
        model = UserModel.query.get(int(user.id_))
        if model:
            model.username = user.username
            model.display_name = user.display_name
            model.avatar_url = user.avatar_url
            model.is_active = user.is_active
            model.is_verified = user.is_verified
            model.last_login = user.last_login
            db.session.commit()
        return self._to_domain(model)

    def get_model_by_email(self, email: str) -> UserModel | None:
        """Get raw model for authentication (includes password_hash)."""
        return UserModel.query.filter_by(email=email.lower().strip()).first()

    def _to_domain(self, model: UserModel) -> User:
        """Convert ORM model to domain entity."""
        return User(
            id_=UUID(int=model.id),
            email=model.email,
            username=model.username,
            display_name=model.display_name,
            avatar_url=model.avatar_url,
            is_active=model.is_active,
            is_verified=model.is_verified,
            created_at=model.created_at,
            updated_at=model.updated_at,
            last_login=model.last_login,
        )


class SQLTransferJobRepository(TransferJobRepository):
    """SQLAlchemy implementation of TransferJobRepository port."""

    def get_by_id(self, job_id: UUID) -> TransferJob | None:
        model = TransferJobModel.query.get(int(job_id))
        if not model:
            return None
        return self._to_domain(model)

    def get_by_user(self, user_id: UUID, limit: int = 10) -> list[TransferJob]:
        models = (
            TransferJobModel.query
            .filter_by(user_id=int(user_id))
            .order_by(TransferJobModel.created_at.desc())
            .limit(limit)
            .all()
        )
        return [self._to_domain(m) for m in models]

    def create(self, job: TransferJob) -> TransferJob:
        model = TransferJobModel(
            user_id=int(job.user_id),
            method=job.params.method.value,
            parameters=job.params.model_dump(),
            status=job.status.value,
            error_message=job.error_message,
            processing_time_ms=job.processing_time_ms,
        )
        db.session.add(model)
        db.session.commit()

        # Update job with generated ID
        job.id_ = UUID(int=model.id)
        return job

    def update(self, job: TransferJob) -> TransferJob:
        model = TransferJobModel.query.get(int(job.id_))
        if model:
            model.status = job.status.value
            model.error_message = job.error_message
            model.completed_at = job.completed_at
            model.processing_time_ms = job.processing_time_ms
            db.session.commit()
        return job

    def _to_domain(self, model: TransferJobModel) -> TransferJob:
        """Convert ORM model to domain entity."""
        params = TransferParams(**(model.parameters or {}))
        return TransferJob(
            id_=UUID(int=model.id),
            user_id=UUID(int=model.user_id),
            params=params,
            status=TransferStatus(model.status),
            error_message=model.error_message,
            created_at=model.created_at,
            completed_at=model.completed_at,
            processing_time_ms=model.processing_time_ms,
        )
