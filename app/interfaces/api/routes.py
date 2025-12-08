# ABOUTME: REST API routes - thin controllers for palette transfer operations
# ABOUTME: Handles HTTP, delegates business logic to services

import base64
from io import BytesIO

import numpy as np
from PIL import Image
from flask import request, jsonify
import httpx

from app.interfaces.api import bp
from app.interfaces.api.deps import get_transfer_service_stateless
from app.domain.transfer import TransferParams, SkinDetectionParams
from app.domain.exceptions import ImageProcessingError, ValidationError


def _get_image_from_request(field_name: str) -> np.ndarray:
    """Extract image from multipart file upload, base64, or URL.

    Args:
        field_name: Name of the field (e.g., 'source', 'target', 'image')

    Returns:
        RGB image as numpy array

    Raises:
        ValueError: If no image provided
    """
    # File upload
    if field_name in request.files:
        file = request.files[field_name]
        if file.filename:
            image = Image.open(file.stream).convert("RGB")
            return np.array(image)

    # JSON body
    if request.is_json:
        data = request.get_json()
        if f"{field_name}_base64" in data:
            return _base64_to_image(data[f"{field_name}_base64"])
        if f"{field_name}_url" in data:
            return _fetch_image(data[f"{field_name}_url"])

    # Form data
    if f"{field_name}_base64" in request.form:
        return _base64_to_image(request.form[f"{field_name}_base64"])
    if f"{field_name}_url" in request.form:
        return _fetch_image(request.form[f"{field_name}_url"])

    raise ValueError(f"No image provided for {field_name}")


def _base64_to_image(b64_string: str) -> np.ndarray:
    """Convert base64 string to numpy array."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    image_data = base64.b64decode(b64_string)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    return np.array(image)


def _fetch_image(url: str) -> np.ndarray:
    """Fetch image from URL."""
    response = httpx.get(url, follow_redirects=True, timeout=10.0)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(image)


def _get_params_from_request() -> dict:
    """Extract parameters from JSON or form data."""
    if request.is_json:
        return request.get_json()
    return dict(request.form)


@bp.route("/detect", methods=["POST"])
def detect_skin():
    """Detect skin regions in an image.

    Accepts multipart file upload or JSON with base64/URL.
    Returns skin mask and statistics.
    """
    try:
        image = _get_image_from_request("image")
        params = _get_params_from_request()

        detection_params = SkinDetectionParams(
            cr_low=int(params.get("cr_low", 133)),
            cr_high=int(params.get("cr_high", 173)),
            cb_low=int(params.get("cb_low", 77)),
            cb_high=int(params.get("cb_high", 127)),
        )

        service = get_transfer_service_stateless()
        result = service.detect_skin(image, detection_params)

        return jsonify({
            "success": True,
            "skin_pixels": result.skin_pixels,
            "total_pixels": result.total_pixels,
            "skin_percentage": result.skin_percentage,
            "visualization": result.visualization_base64,
            "skin_mask": result.skin_mask_base64,
            "detection_params": result.detection_params.model_dump(),
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/transfer", methods=["POST"])
def transfer():
    """Transfer palette from source to target image.

    Methods: skintone, reinhard, hybrid, optimized
    """
    try:
        source = _get_image_from_request("source")
        target = _get_image_from_request("target")
        params_dict = _get_params_from_request()

        # Build TransferParams from request
        transfer_params = TransferParams(
            method=params_dict.get("method", "skintone"),
            skin_blend=float(params_dict.get("skin_blend", 0.9)),
            hair_blend=float(params_dict.get("hair_blend", 0.5)),
            bg_blend=float(params_dict.get("bg_blend", 0.3)),
            preserve_luminance=str(params_dict.get("preserve_luminance", "false")).lower() == "true",
            skin_weight=float(params_dict.get("skin_weight", 0.7)),
            skin_detection=SkinDetectionParams(
                cr_low=int(params_dict.get("cr_low", 133)),
                cr_high=int(params_dict.get("cr_high", 173)),
                cb_low=int(params_dict.get("cb_low", 77)),
                cb_high=int(params_dict.get("cb_high", 127)),
            ),
        )

        service = get_transfer_service_stateless()
        result = service.execute_transfer(source, target, transfer_params)

        return jsonify({
            "success": True,
            "result": result.result_base64,
            "method": transfer_params.method.value,
            "processing_time_ms": result.job.processing_time_ms,
            "source_skin_pixels": result.source_skin_pixels,
            "target_skin_pixels": result.target_skin_pixels,
        })

    except (ValueError, ValidationError) as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImageProcessingError as e:
        return jsonify({"success": False, "error": str(e)}), 422
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/compare", methods=["POST"])
def compare():
    """Compare all transfer methods side by side.

    Returns original + all method results.
    """
    try:
        source = _get_image_from_request("source")
        target = _get_image_from_request("target")

        service = get_transfer_service_stateless()
        results = service.compare_methods(source, target)

        # Format response
        formatted = {
            "original": service._image_to_base64(target),
        }

        for method, result in results.items():
            if isinstance(result, str):
                formatted[method] = {"error": result}
            else:
                formatted[method] = {
                    "image": result.result_base64,
                    "time_ms": result.job.processing_time_ms,
                }

        return jsonify({
            "success": True,
            "results": formatted,
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})
