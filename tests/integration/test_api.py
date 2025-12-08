# ABOUTME: Integration tests for REST API endpoints
# ABOUTME: Tests /api/detect, /api/transfer, /api/compare with real Flask app

import pytest
import base64
import numpy as np
from io import BytesIO
from PIL import Image

from app import create_app


@pytest.fixture
def app():
    """Create test Flask app."""
    app = create_app("testing")
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def sample_image_base64():
    """Create a sample base64 image for testing."""
    # Create a 50x50 RGB image
    image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@pytest.fixture
def sample_image_file():
    """Create a sample image file for testing."""
    image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class TestHealthEndpoint:
    """Test health check endpoints."""

    def test_health_returns_ok(self, client):
        """Test /health returns ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json["status"] == "ok"

    def test_api_health_returns_ok(self, client):
        """Test /api/health returns ok."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json["status"] == "ok"


class TestDetectEndpoint:
    """Test /api/detect endpoint."""

    def test_detect_with_base64(self, client, sample_image_base64):
        """Test skin detection with base64 image."""
        response = client.post(
            "/api/detect",
            json={"image_base64": sample_image_base64},
        )

        assert response.status_code == 200
        data = response.json
        assert data["success"] is True
        assert "skin_pixels" in data
        assert "total_pixels" in data
        assert "skin_percentage" in data
        assert "visualization" in data
        assert data["visualization"].startswith("data:image/")

    def test_detect_with_file_upload(self, client, sample_image_file):
        """Test skin detection with file upload."""
        response = client.post(
            "/api/detect",
            data={"image": (sample_image_file, "test.png")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        data = response.json
        assert data["success"] is True

    def test_detect_with_custom_params(self, client, sample_image_base64):
        """Test skin detection with custom detection params."""
        response = client.post(
            "/api/detect",
            json={
                "image_base64": sample_image_base64,
                "cr_low": 140,
                "cr_high": 170,
            },
        )

        assert response.status_code == 200
        data = response.json
        assert data["detection_params"]["cr_low"] == 140
        assert data["detection_params"]["cr_high"] == 170

    def test_detect_no_image_returns_error(self, client):
        """Test detect without image returns error."""
        response = client.post("/api/detect", json={})
        assert response.status_code == 400
        assert response.json["success"] is False


class TestTransferEndpoint:
    """Test /api/transfer endpoint."""

    def test_transfer_with_base64(self, client, sample_image_base64):
        """Test transfer with base64 images."""
        response = client.post(
            "/api/transfer",
            json={
                "source_base64": sample_image_base64,
                "target_base64": sample_image_base64,
            },
        )

        assert response.status_code == 200
        data = response.json
        assert data["success"] is True
        assert "result" in data
        assert data["result"].startswith("data:image/")
        assert "processing_time_ms" in data

    def test_transfer_with_file_upload(self, client, sample_image_file):
        """Test transfer with file uploads."""
        source = BytesIO(sample_image_file.getvalue())
        target = BytesIO(sample_image_file.getvalue())

        response = client.post(
            "/api/transfer",
            data={
                "source": (source, "source.png"),
                "target": (target, "target.png"),
            },
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        data = response.json
        assert data["success"] is True

    def test_transfer_skintone_method(self, client, sample_image_base64):
        """Test transfer with skintone method."""
        response = client.post(
            "/api/transfer",
            json={
                "source_base64": sample_image_base64,
                "target_base64": sample_image_base64,
                "method": "skintone",
                "skin_blend": 0.8,
            },
        )

        assert response.status_code == 200
        data = response.json
        assert data["method"] == "skintone"

    def test_transfer_reinhard_method(self, client, sample_image_base64):
        """Test transfer with reinhard method."""
        response = client.post(
            "/api/transfer",
            json={
                "source_base64": sample_image_base64,
                "target_base64": sample_image_base64,
                "method": "reinhard",
            },
        )

        assert response.status_code == 200
        data = response.json
        assert data["method"] == "reinhard"

    def test_transfer_missing_source_returns_error(self, client, sample_image_base64):
        """Test transfer without source returns error."""
        response = client.post(
            "/api/transfer",
            json={"target_base64": sample_image_base64},
        )

        assert response.status_code == 400
        assert response.json["success"] is False


class TestCompareEndpoint:
    """Test /api/compare endpoint."""

    def test_compare_returns_all_methods(self, client, sample_image_base64):
        """Test compare returns results for all methods."""
        response = client.post(
            "/api/compare",
            json={
                "source_base64": sample_image_base64,
                "target_base64": sample_image_base64,
            },
        )

        assert response.status_code == 200
        data = response.json
        assert data["success"] is True
        assert "results" in data

        results = data["results"]
        assert "original" in results
        assert "skintone" in results
        assert "reinhard" in results
        assert "hybrid" in results
        assert "optimized" in results

    def test_compare_includes_timing(self, client, sample_image_base64):
        """Test compare includes timing for each method."""
        response = client.post(
            "/api/compare",
            json={
                "source_base64": sample_image_base64,
                "target_base64": sample_image_base64,
            },
        )

        assert response.status_code == 200
        data = response.json

        for method in ["skintone", "reinhard", "hybrid", "optimized"]:
            if "image" in data["results"][method]:
                assert "time_ms" in data["results"][method]
