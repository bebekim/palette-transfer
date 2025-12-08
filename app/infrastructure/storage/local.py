# ABOUTME: Local filesystem image storage implementation
# ABOUTME: Implements ImageStorage port for local development

import os
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.ports import ImageStorage


class LocalImageStorage(ImageStorage):
    """Local filesystem implementation of ImageStorage port."""

    def __init__(self, base_path: str = "uploads"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, image: np.ndarray, path: str) -> str:
        """Save image to local filesystem.

        Args:
            image: RGB image as numpy array
            path: Relative path within base_path

        Returns:
            Full path to saved image
        """
        full_path = self.base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_image.save(str(full_path), quality=90)

        return str(full_path)

    def load(self, path: str) -> np.ndarray:
        """Load image from local filesystem.

        Args:
            path: Path to image (relative to base_path or absolute)

        Returns:
            RGB image as numpy array
        """
        if os.path.isabs(path):
            full_path = Path(path)
        else:
            full_path = self.base_path / path

        image = Image.open(full_path).convert("RGB")
        return np.array(image)

    def delete(self, path: str) -> bool:
        """Delete image from local filesystem.

        Args:
            path: Path to image

        Returns:
            True if deleted, False if not found
        """
        if os.path.isabs(path):
            full_path = Path(path)
        else:
            full_path = self.base_path / path

        if full_path.exists():
            full_path.unlink()
            return True
        return False

    def get_url(self, path: str) -> str:
        """Get URL for image (for local, just returns path).

        Args:
            path: Path to image

        Returns:
            URL or path to access image
        """
        return f"/uploads/{path}"
