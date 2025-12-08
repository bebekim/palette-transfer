# ABOUTME: Domain-specific exceptions for business rule violations
# ABOUTME: These are raised by domain/service layers, caught by interface layer


class DomainError(Exception):
    """Base exception for all domain errors."""

    def __init__(self, message: str = "A domain error occurred"):
        self.message = message
        super().__init__(self.message)


class ValidationError(DomainError):
    """Raised when input validation fails."""

    def __init__(self, message: str = "Validation failed", field: str | None = None):
        self.field = field
        super().__init__(message)


class ImageProcessingError(DomainError):
    """Raised when image processing fails."""

    def __init__(self, message: str = "Image processing failed"):
        super().__init__(message)


class UserNotFoundError(DomainError):
    """Raised when user lookup fails."""

    def __init__(self, user_id: str | None = None):
        message = f"User not found: {user_id}" if user_id else "User not found"
        super().__init__(message)


class TransferJobNotFoundError(DomainError):
    """Raised when transfer job lookup fails."""

    def __init__(self, job_id: str | None = None):
        message = f"Transfer job not found: {job_id}" if job_id else "Transfer job not found"
        super().__init__(message)
