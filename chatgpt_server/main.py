# -*- coding: utf-8 -*-
"""
ChatGPT MCP Server for Palette Transfer

Exposes skin-tone transfer algorithms as MCP tools that ChatGPT can invoke.
"""

import os
import sys
import base64
import tempfile
from io import BytesIO
from typing import Optional

import httpx
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from fastmcp import FastMCP

# Add algorithms to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'algorithms'))

from skintone_transfer_pil import SkinToneTransfer


# Initialize MCP server
mcp = FastMCP("Palette Transfer")


class SkinToneTransferInput(BaseModel):
    """Input schema for skin-tone transfer tool."""
    source_url: str = Field(
        description="URL of the source image (the skin tone to copy FROM)"
    )
    target_url: str = Field(
        description="URL of the target image (the image to apply skin tone TO)"
    )
    skin_blend: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Blending factor for skin regions (0.0-1.0)"
    )
    hair_blend: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Blending factor for hair regions (0.0-1.0)"
    )
    bg_blend: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Blending factor for background (0.0-1.0)"
    )
    skin_cr_low: int = Field(
        default=133,
        ge=0,
        le=255,
        description="Lower Cr bound for skin detection (use ~120 for darker skin)"
    )
    skin_cr_high: int = Field(
        default=173,
        ge=0,
        le=255,
        description="Upper Cr bound for skin detection (use ~180 for lighter skin)"
    )
    skin_cb_low: int = Field(
        default=77,
        ge=0,
        le=255,
        description="Lower Cb bound for skin detection"
    )
    skin_cb_high: int = Field(
        default=127,
        ge=0,
        le=255,
        description="Upper Cb bound for skin detection (use ~140 for darker skin)"
    )


class DetectSkinInput(BaseModel):
    """Input schema for skin detection tool."""
    image_url: str = Field(
        description="URL of the image to detect skin regions in"
    )
    skin_cr_low: int = Field(default=133, ge=0, le=255)
    skin_cr_high: int = Field(default=173, ge=0, le=255)
    skin_cb_low: int = Field(default=77, ge=0, le=255)
    skin_cb_high: int = Field(default=127, ge=0, le=255)


async def fetch_image(url: str) -> np.ndarray:
    """Fetch an image from URL and return as RGB numpy array."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert('RGB')
    return np.array(image)


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 PNG string."""
    pil_image = Image.fromarray(image.astype(np.uint8))
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def mask_to_base64(mask: np.ndarray) -> str:
    """Convert mask (0-1 float) to base64 PNG string."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    pil_image = Image.fromarray(mask_uint8, mode='L')
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@mcp.tool()
async def transfer_skin_tone(
    source_url: str,
    target_url: str,
    skin_blend: float = 0.9,
    hair_blend: float = 0.5,
    bg_blend: float = 0.3,
    skin_cr_low: int = 133,
    skin_cr_high: int = 173,
    skin_cb_low: int = 77,
    skin_cb_high: int = 127,
) -> dict:
    """
    Transfer skin tone from a source image to a target image.

    The algorithm:
    1. Detects skin pixels in the source image using YCrCb color space
    2. Computes LAB color statistics from only those skin pixels
    3. Applies Reinhard color transfer to the target's skin regions

    This ensures the palette is derived only from skin, not hair or background.

    Args:
        source_url: URL of the source image (skin tone to copy FROM)
        target_url: URL of the target image (image to apply skin tone TO)
        skin_blend: How much to apply transfer to skin (0.0-1.0)
        hair_blend: How much to apply transfer to hair (0.0-1.0)
        bg_blend: How much to apply transfer to background (0.0-1.0)
        skin_cr_low: Lower Cr bound for skin detection (lower for darker skin)
        skin_cr_high: Upper Cr bound for skin detection (higher for lighter skin)
        skin_cb_low: Lower Cb bound for skin detection
        skin_cb_high: Upper Cb bound for skin detection

    Returns:
        Dictionary with result image and diagnostic info
    """
    try:
        # Fetch images
        source_image = await fetch_image(source_url)
        target_image = await fetch_image(target_url)

        # Initialize transfer
        transfer = SkinToneTransfer(
            skin_blend_factor=skin_blend,
            hair_region_blend_factor=hair_blend,
            background_blend_factor=bg_blend,
            skin_ycrcb_lower=(0, skin_cr_low, skin_cb_low),
            skin_ycrcb_upper=(255, skin_cr_high, skin_cb_high),
        )

        # Perform transfer
        transfer.fit(source_image)
        result = transfer.recolor(target_image)

        # Prepare response
        response = {
            "success": True,
            "result_image": f"data:image/png;base64,{image_to_base64(result)}",
            "source_skin_pixels": int(np.sum(transfer.source_skin_mask > 0.5)),
            "target_skin_pixels": int(np.sum(transfer.target_skin_mask > 0.5)),
            "source_lab_mean": {
                "L": float(transfer.source_mean[0]),
                "a": float(transfer.source_mean[1]),
                "b": float(transfer.source_mean[2]),
            },
            "source_lab_std": {
                "L": float(transfer.source_std[0]),
                "a": float(transfer.source_std[1]),
                "b": float(transfer.source_std[2]),
            },
        }

        # Include masks for visualization
        if transfer.source_skin_mask is not None:
            response["source_skin_mask"] = f"data:image/png;base64,{mask_to_base64(transfer.source_skin_mask)}"
        if transfer.target_skin_mask is not None:
            response["target_skin_mask"] = f"data:image/png;base64,{mask_to_base64(transfer.target_skin_mask)}"

        return response

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def detect_skin_regions(
    image_url: str,
    skin_cr_low: int = 133,
    skin_cr_high: int = 173,
    skin_cb_low: int = 77,
    skin_cb_high: int = 127,
) -> dict:
    """
    Detect and visualize skin regions in an image.

    Uses YCrCb color space thresholding combined with face detection
    to identify skin pixels. Useful for tuning detection parameters.

    Args:
        image_url: URL of the image to analyze
        skin_cr_low: Lower Cr bound (decrease for darker skin)
        skin_cr_high: Upper Cr bound (increase for lighter skin)
        skin_cb_low: Lower Cb bound
        skin_cb_high: Upper Cb bound (increase for darker skin)

    Returns:
        Dictionary with skin mask and statistics
    """
    try:
        image = await fetch_image(image_url)

        # Use SkinToneTransfer just for mask creation
        transfer = SkinToneTransfer(
            skin_ycrcb_lower=(0, skin_cr_low, skin_cb_low),
            skin_ycrcb_upper=(255, skin_cr_high, skin_cb_high),
        )

        skin_mask, hair_mask = transfer._create_skin_mask(image)

        # Calculate statistics
        skin_pixel_count = int(np.sum(skin_mask > 0.5))
        total_pixels = image.shape[0] * image.shape[1]
        skin_percentage = (skin_pixel_count / total_pixels) * 100

        return {
            "success": True,
            "skin_mask": f"data:image/png;base64,{mask_to_base64(skin_mask)}",
            "hair_region_mask": f"data:image/png;base64,{mask_to_base64(hair_mask)}",
            "skin_pixel_count": skin_pixel_count,
            "total_pixels": total_pixels,
            "skin_percentage": round(skin_percentage, 2),
            "detection_bounds": {
                "cr_range": [skin_cr_low, skin_cr_high],
                "cb_range": [skin_cb_low, skin_cb_high],
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@mcp.tool()
async def compare_before_after(
    source_url: str,
    target_url: str,
    skin_blend: float = 0.9,
    hair_blend: float = 0.5,
    bg_blend: float = 0.3,
    skin_cr_low: int = 133,
    skin_cr_high: int = 173,
    skin_cb_low: int = 77,
    skin_cb_high: int = 127,
) -> dict:
    """
    Create a side-by-side comparison of before and after skin tone transfer.

    Returns the original target image alongside the transferred result
    for easy visual comparison.

    Args:
        source_url: URL of the source image (skin tone to copy FROM)
        target_url: URL of the target image (image to apply skin tone TO)
        skin_blend: How much to apply transfer to skin (0.0-1.0)
        hair_blend: How much to apply transfer to hair (0.0-1.0)
        bg_blend: How much to apply transfer to background (0.0-1.0)
        skin_cr_low: Lower Cr bound for skin detection
        skin_cr_high: Upper Cr bound for skin detection
        skin_cb_low: Lower Cb bound for skin detection
        skin_cb_high: Upper Cb bound for skin detection

    Returns:
        Dictionary with comparison image and individual images
    """
    try:
        # Fetch images
        source_image = await fetch_image(source_url)
        target_image = await fetch_image(target_url)

        # Initialize transfer
        transfer = SkinToneTransfer(
            skin_blend_factor=skin_blend,
            hair_region_blend_factor=hair_blend,
            background_blend_factor=bg_blend,
            skin_ycrcb_lower=(0, skin_cr_low, skin_cb_low),
            skin_ycrcb_upper=(255, skin_cr_high, skin_cb_high),
        )

        # Perform transfer
        transfer.fit(source_image)
        result = transfer.recolor(target_image)

        # Create side-by-side comparison
        # Resize images to same height if needed
        h1, w1 = target_image.shape[:2]
        h2, w2 = result.shape[:2]

        # Use the smaller height
        target_height = min(h1, h2, 800)  # Cap at 800px for reasonable size

        # Resize both to same height
        scale1 = target_height / h1
        scale2 = target_height / h2

        new_w1 = int(w1 * scale1)
        new_w2 = int(w2 * scale2)

        target_resized = np.array(Image.fromarray(target_image).resize((new_w1, target_height)))
        result_resized = np.array(Image.fromarray(result).resize((new_w2, target_height)))

        # Add labels
        # Create comparison with a small gap
        gap = 10
        comparison = np.ones((target_height, new_w1 + gap + new_w2, 3), dtype=np.uint8) * 255
        comparison[:, :new_w1] = target_resized
        comparison[:, new_w1 + gap:] = result_resized

        response = {
            "success": True,
            "comparison_image": f"data:image/png;base64,{image_to_base64(comparison)}",
            "before_image": f"data:image/png;base64,{image_to_base64(target_image)}",
            "after_image": f"data:image/png;base64,{image_to_base64(result)}",
            "source_skin_pixels": int(np.sum(transfer.source_skin_mask > 0.5)),
            "target_skin_pixels": int(np.sum(transfer.target_skin_mask > 0.5)),
            "transfer_stats": {
                "source_lab_mean": {
                    "L": float(transfer.source_mean[0]),
                    "a": float(transfer.source_mean[1]),
                    "b": float(transfer.source_mean[2]),
                },
                "source_lab_std": {
                    "L": float(transfer.source_std[0]),
                    "a": float(transfer.source_std[1]),
                    "b": float(transfer.source_std[2]),
                },
            },
        }

        return response

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# Create the ASGI app
app = mcp.http_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
