# ABOUTME: SQLAlchemy ORM models - database representation
# ABOUTME: These map to database tables, separate from domain entities

from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from app.extensions import db, login_manager


class UserModel(UserMixin, db.Model):
    """User database model."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    username = db.Column(db.String(100), unique=True, nullable=True, index=True)
    password_hash = db.Column(db.String(255), nullable=True)  # Nullable for OAuth-only users

    # OAuth
    oauth_provider = db.Column(db.String(50), nullable=True)  # 'google', etc.
    oauth_id = db.Column(db.String(255), nullable=True, index=True)  # Provider's user ID

    # Profile
    display_name = db.Column(db.String(100), nullable=True)
    avatar_url = db.Column(db.Text, nullable=True)

    # Status
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    # Relationships
    transfers = db.relationship("TransferJobModel", backref="user", lazy="dynamic")

    def set_password(self, password: str) -> None:
        """Hash and set password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify password."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"<UserModel {self.email}>"


class TransferJobModel(db.Model):
    """Transfer job database model."""

    __tablename__ = "transfer_jobs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)

    # Transfer settings
    method = db.Column(db.String(50), nullable=False)
    parameters = db.Column(db.JSON, nullable=True)

    # Status
    status = db.Column(db.String(20), default="pending")
    error_message = db.Column(db.Text, nullable=True)

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    # Processing stats
    processing_time_ms = db.Column(db.Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<TransferJobModel {self.id} ({self.status})>"


@login_manager.user_loader
def load_user(user_id: int) -> UserModel | None:
    """Load user by ID for Flask-Login."""
    return UserModel.query.get(int(user_id))
