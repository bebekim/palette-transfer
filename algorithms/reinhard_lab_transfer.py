# -*- coding: utf-8 -*-
"""
Reinhard Lab Transfer - Classic statistical color transfer.

Based on: Reinhard et al. (2001) "Color Transfer between Images"

Performance: ~100-120ms for full resolution
Quality: Excellent for general color matching

This is the recommended default method for most use cases.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from PIL import Image
from scipy import ndimage


@dataclass
class LabStats:
    """LAB channel statistics."""
    mean: np.ndarray  # [L, a, b] means
    std: np.ndarray   # [L, a, b] standard deviations


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to LAB color space.

    Args:
        rgb: RGB image array (H, W, 3) with values 0-255

    Returns:
        LAB image array (H, W, 3) scaled to 0-255 range
    """
    # Normalize RGB to 0-1
    rgb_norm = rgb.astype(np.float32) / 255.0

    # Convert to linear RGB (remove gamma)
    mask = rgb_norm > 0.04045
    rgb_linear = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

    # RGB to XYZ (D65 illuminant)
    r, g, b = rgb_linear[:, :, 0], rgb_linear[:, :, 1], rgb_linear[:, :, 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Normalize for D65
    x = x / 0.95047
    z = z / 1.08883

    # XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3

    fx = np.where(x > epsilon, np.cbrt(x), (kappa * x + 16) / 116)
    fy = np.where(y > epsilon, np.cbrt(y), (kappa * y + 16) / 116)
    fz = np.where(z > epsilon, np.cbrt(z), (kappa * z + 16) / 116)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ch = 200 * (fy - fz)

    # Scale to 0-255 range
    L = L * 255 / 100
    a = a + 128
    b_ch = b_ch + 128

    lab = np.stack([L, a, b_ch], axis=2)
    return np.clip(lab, 0, 255).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB image to RGB color space.

    Args:
        lab: LAB image array (H, W, 3) with values 0-255

    Returns:
        RGB image array (H, W, 3)
    """
    # Unscale from 0-255 range
    L = lab[:, :, 0].astype(np.float32) * 100 / 255
    a = lab[:, :, 1].astype(np.float32) - 128
    b_ch = lab[:, :, 2].astype(np.float32) - 128

    # LAB to XYZ
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b_ch / 200

    epsilon = 0.008856
    kappa = 903.3

    x = np.where(fx ** 3 > epsilon, fx ** 3, (116 * fx - 16) / kappa)
    y = np.where(L > kappa * epsilon, ((L + 16) / 116) ** 3, L / kappa)
    z = np.where(fz ** 3 > epsilon, fz ** 3, (116 * fz - 16) / kappa)

    # Denormalize for D65
    x = x * 0.95047
    z = z * 1.08883

    # XYZ to linear RGB
    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    # Clip negatives
    r = np.clip(r, 0, None)
    g = np.clip(g, 0, None)
    b = np.clip(b, 0, None)

    # Apply gamma
    mask_r = r > 0.0031308
    mask_g = g > 0.0031308
    mask_b = b > 0.0031308

    r = np.where(mask_r, 1.055 * (r ** (1/2.4)) - 0.055, 12.92 * r)
    g = np.where(mask_g, 1.055 * (g ** (1/2.4)) - 0.055, 12.92 * g)
    b = np.where(mask_b, 1.055 * (b ** (1/2.4)) - 0.055, 12.92 * b)

    rgb = np.stack([r, g, b], axis=2) * 255
    return np.clip(rgb, 0, 255).astype(np.uint8)


def compute_lab_stats(image: np.ndarray, max_size: int = 512) -> LabStats:
    """Compute LAB statistics efficiently at reduced resolution.

    Args:
        image: RGB image array
        max_size: Maximum dimension for stats computation

    Returns:
        LabStats with mean and std for each channel
    """
    h, w = image.shape[:2]

    # Downsample for efficiency
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image_small = np.array(Image.fromarray(image).resize((new_w, new_h)))
    else:
        image_small = image

    lab = rgb_to_lab(image_small)

    return LabStats(
        mean=lab.mean(axis=(0, 1)),
        std=lab.std(axis=(0, 1))
    )


class ReinhardLabTransfer:
    """
    Classic Reinhard Lab Transfer.

    Transfers color statistics from reference image to source image
    using mean and standard deviation matching in LAB color space.

    Usage:
        transfer = ReinhardLabTransfer()
        transfer.fit(reference_image)
        result = transfer.apply(source_image)
    """

    def __init__(self, preserve_luminance: bool = False, stats_size: int = 512):
        """
        Args:
            preserve_luminance: If True, only transfer chrominance (a, b channels)
            stats_size: Resolution for computing statistics (larger = more accurate but slower)
        """
        self.preserve_luminance = preserve_luminance
        self.stats_size = stats_size
        self.reference_stats: Optional[LabStats] = None

    def fit(self, reference: np.ndarray) -> 'ReinhardLabTransfer':
        """
        Learn color statistics from reference image.

        Args:
            reference: Reference RGB image (the "goal" colors)

        Returns:
            self for method chaining
        """
        self.reference_stats = compute_lab_stats(reference, self.stats_size)
        return self

    def apply(self, source: np.ndarray) -> np.ndarray:
        """
        Apply color transfer to source image.

        Args:
            source: Source RGB image to transform

        Returns:
            Transformed RGB image with colors matching reference
        """
        if self.reference_stats is None:
            raise ValueError("Must call fit() before apply()")

        # Compute source stats
        source_stats = compute_lab_stats(source, self.stats_size)

        # Convert full source to LAB
        source_lab = rgb_to_lab(source)

        # Compute transfer parameters
        scale = self.reference_stats.std / (source_stats.std + 1e-6)
        shift = self.reference_stats.mean - source_stats.mean * scale

        if self.preserve_luminance:
            scale[0] = 1.0
            shift[0] = 0.0

        # Apply transfer
        result_lab = source_lab * scale + shift
        result_lab = np.clip(result_lab, 0, 255)

        return lab_to_rgb(result_lab)

    def fit_apply(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Convenience method to fit and apply in one call.

        Args:
            source: Source RGB image to transform
            reference: Reference RGB image (the "goal" colors)

        Returns:
            Transformed RGB image
        """
        return self.fit(reference).apply(source)


# Convenience functions
def transfer_reinhard(source: np.ndarray, reference: np.ndarray,
                      preserve_luminance: bool = False) -> np.ndarray:
    """
    Quick Reinhard color transfer.

    Args:
        source: Source RGB image to transform
        reference: Reference RGB image (the "goal" colors)
        preserve_luminance: If True, only transfer chrominance

    Returns:
        Transformed RGB image
    """
    return ReinhardLabTransfer(preserve_luminance=preserve_luminance).fit_apply(source, reference)
