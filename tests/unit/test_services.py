# ABOUTME: Unit tests for service layer
# ABOUTME: Tests TransferService with mocked dependencies

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from uuid import uuid4

from app.domain.transfer import (
    TransferMethod,
    TransferParams,
    SkinDetectionParams,
    TransferStatus,
)
from app.services.transfer_service import TransferService


class TestTransferServiceDetectSkin:
    """Test TransferService.detect_skin method."""

    def test_detect_skin_returns_result(self):
        """Test skin detection returns valid result."""
        service = TransferService()

        # Create a simple test image (100x100 RGB)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = service.detect_skin(image)

        assert result.total_pixels == 10000
        assert 0 <= result.skin_percentage <= 100
        assert result.visualization_base64.startswith("data:image/")
        assert result.skin_mask_base64.startswith("data:image/")

    def test_detect_skin_with_custom_params(self):
        """Test skin detection with custom parameters."""
        service = TransferService()

        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        params = SkinDetectionParams(cr_low=140, cr_high=170)

        result = service.detect_skin(image, params)

        assert result.detection_params.cr_low == 140
        assert result.detection_params.cr_high == 170


class TestTransferServiceExecuteTransfer:
    """Test TransferService.execute_transfer method."""

    def test_execute_transfer_skintone(self):
        """Test skintone transfer execution."""
        service = TransferService()

        source = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        params = TransferParams(method=TransferMethod.SKINTONE)

        result = service.execute_transfer(source, target, params)

        assert result.result_base64.startswith("data:image/")
        assert result.job.status == TransferStatus.COMPLETED
        assert result.job.processing_time_ms is not None
        assert result.job.processing_time_ms >= 0

    def test_execute_transfer_reinhard(self):
        """Test reinhard transfer execution."""
        service = TransferService()

        source = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        params = TransferParams(method=TransferMethod.REINHARD)

        result = service.execute_transfer(source, target, params)

        assert result.result_base64.startswith("data:image/")
        assert result.job.status == TransferStatus.COMPLETED

    def test_execute_transfer_hybrid(self):
        """Test hybrid transfer execution."""
        service = TransferService()

        source = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        params = TransferParams(method=TransferMethod.HYBRID)

        result = service.execute_transfer(source, target, params)

        assert result.result_base64.startswith("data:image/")

    def test_execute_transfer_optimized(self):
        """Test optimized transfer execution."""
        service = TransferService()

        source = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        params = TransferParams(method=TransferMethod.OPTIMIZED)

        result = service.execute_transfer(source, target, params)

        assert result.result_base64.startswith("data:image/")

    def test_execute_transfer_with_custom_blend(self):
        """Test transfer with custom blend factors."""
        service = TransferService()

        source = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        params = TransferParams(
            method=TransferMethod.SKINTONE,
            skin_blend=0.5,
            hair_blend=0.3,
            bg_blend=0.1,
        )

        result = service.execute_transfer(source, target, params)

        assert result.result_base64.startswith("data:image/")

    def test_execute_transfer_tracks_user(self):
        """Test transfer with user_id."""
        service = TransferService()
        user_id = uuid4()

        source = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        params = TransferParams()

        result = service.execute_transfer(source, target, params, user_id=user_id)

        assert result.job.user_id == user_id


class TestTransferServiceCompareMethods:
    """Test TransferService.compare_methods method."""

    def test_compare_methods_returns_all(self):
        """Test compare returns results for all methods."""
        service = TransferService()

        source = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        target = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)

        results = service.compare_methods(source, target)

        assert TransferMethod.SKINTONE.value in results
        assert TransferMethod.REINHARD.value in results
        assert TransferMethod.HYBRID.value in results
        assert TransferMethod.OPTIMIZED.value in results


class TestTransferServiceImageConversion:
    """Test TransferService image conversion helpers."""

    def test_image_to_base64(self):
        """Test image to base64 conversion."""
        service = TransferService()

        image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        b64 = service._image_to_base64(image)

        assert b64.startswith("data:image/png;base64,")
        assert len(b64) > 50  # Should have actual content
