# ABOUTME: Authentication infrastructure - password hashing implementation
# ABOUTME: Implements PasswordHasher port using werkzeug

from werkzeug.security import generate_password_hash, check_password_hash

from app.services.ports import PasswordHasher


class WerkzeugPasswordHasher(PasswordHasher):
    """Werkzeug implementation of PasswordHasher port."""

    def hash(self, password: str) -> str:
        """Hash a password using werkzeug."""
        return generate_password_hash(password)

    def verify(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return check_password_hash(hashed, password)
