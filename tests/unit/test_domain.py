# ABOUTME: Unit tests for domain entities
# ABOUTME: Tests Pydantic validation and business rules

import pytest
from uuid import UUID
from pydantic import ValidationError

from app.domain.transfer import (
    TransferMethod,
    TransferStatus,
    TransferParams,
    SkinDetectionParams,
    TransferJob,
)
from app.domain.user import User, UserCreate
from app.domain.exceptions import DomainError, ImageProcessingError


class TestTransferParams:
    """Test TransferParams domain entity."""

    def test_default_values(self):
        """Test default parameter values."""
        params = TransferParams()

        assert params.method == TransferMethod.SKINTONE
        assert params.skin_blend == 0.9
        assert params.hair_blend == 0.5
        assert params.bg_blend == 0.3
        assert params.preserve_luminance is False

    def test_valid_blend_factors(self):
        """Test valid blend factor ranges."""
        params = TransferParams(
            skin_blend=0.0,
            hair_blend=0.5,
            bg_blend=1.0,
        )

        assert params.skin_blend == 0.0
        assert params.hair_blend == 0.5
        assert params.bg_blend == 1.0

    def test_invalid_blend_factor_too_high(self):
        """Test that blend factors > 1.0 are rejected."""
        with pytest.raises(ValidationError):
            TransferParams(skin_blend=1.5)

    def test_invalid_blend_factor_negative(self):
        """Test that negative blend factors are rejected."""
        with pytest.raises(ValidationError):
            TransferParams(hair_blend=-0.1)

    def test_all_methods_valid(self):
        """Test all transfer methods can be set."""
        for method in TransferMethod:
            params = TransferParams(method=method)
            assert params.method == method

    def test_skin_detection_params_embedded(self):
        """Test skin detection params are properly embedded."""
        params = TransferParams(
            skin_detection=SkinDetectionParams(
                cr_low=130,
                cr_high=175,
            )
        )

        assert params.skin_detection.cr_low == 130
        assert params.skin_detection.cr_high == 175


class TestSkinDetectionParams:
    """Test SkinDetectionParams domain entity."""

    def test_default_values(self):
        """Test default YCrCb bounds."""
        params = SkinDetectionParams()

        assert params.cr_low == 133
        assert params.cr_high == 173
        assert params.cb_low == 77
        assert params.cb_high == 127

    def test_valid_range(self):
        """Test valid range 0-255."""
        params = SkinDetectionParams(
            cr_low=0,
            cr_high=255,
            cb_low=0,
            cb_high=255,
        )

        assert params.cr_low == 0
        assert params.cr_high == 255

    def test_invalid_range_negative(self):
        """Test negative values rejected."""
        with pytest.raises(ValidationError):
            SkinDetectionParams(cr_low=-1)

    def test_invalid_range_too_high(self):
        """Test values > 255 rejected."""
        with pytest.raises(ValidationError):
            SkinDetectionParams(cb_high=256)


class TestTransferJob:
    """Test TransferJob domain entity."""

    def test_creates_with_uuid(self):
        """Test job gets UUID on creation."""
        from uuid import uuid4
        user_id = uuid4()

        job = TransferJob(
            user_id=user_id,
            params=TransferParams(),
        )

        assert isinstance(job.id_, UUID)
        assert job.user_id == user_id
        assert job.status == TransferStatus.PENDING

    def test_default_status_pending(self):
        """Test default status is pending."""
        from uuid import uuid4

        job = TransferJob(
            user_id=uuid4(),
            params=TransferParams(),
        )

        assert job.status == TransferStatus.PENDING
        assert job.error_message is None
        assert job.completed_at is None


class TestUserCreate:
    """Test UserCreate domain entity."""

    def test_valid_user_create(self):
        """Test valid user creation data."""
        user = UserCreate(
            email="test@example.com",
            password="securepassword123",
        )

        assert user.email == "test@example.com"
        assert user.password == "securepassword123"

    def test_invalid_email(self):
        """Test invalid email rejected."""
        with pytest.raises(ValidationError):
            UserCreate(email="not-an-email", password="password123")

    def test_password_too_short(self):
        """Test short password rejected."""
        with pytest.raises(ValidationError):
            UserCreate(email="test@example.com", password="short")


class TestDomainExceptions:
    """Test domain exception classes."""

    def test_domain_error_message(self):
        """Test DomainError has message."""
        error = DomainError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_image_processing_error(self):
        """Test ImageProcessingError."""
        error = ImageProcessingError("Transfer failed")
        assert "Transfer failed" in str(error)
        assert isinstance(error, DomainError)
