# -*- coding: utf-8 -*-
# algorithms/skintone_transfer_pil.py
"""
Skin-tone specific color transfer (PIL/NumPy implementation - no OpenCV).

Transfers skin-tone from the source image to the target image by:
1. Detecting skin pixels in the source image
2. Computing LAB color statistics from only those skin pixels
3. Applying Reinhard color transfer to the target's skin regions

This ensures that only skin-tone colors are used for the transfer,
ignoring background, hair, and other non-skin regions in the source.
"""

import time
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def _log_time(start, label):
    """Log elapsed time since start."""
    elapsed = (time.time() - start) * 1000
    print(f"[TIMING] {label}: {elapsed:.1f}ms")
    return time.time()


def rgb_to_ycrcb(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to YCrCb color space.

    Args:
        rgb: RGB image array (H, W, 3) with values 0-255

    Returns:
        YCrCb image array (H, W, 3)
    """
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # ITU-R BT.601 conversion
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128

    ycrcb = np.stack([y, cr, cb], axis=2)
    return np.clip(ycrcb, 0, 255).astype(np.uint8)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to LAB color space.

    Args:
        rgb: RGB image array (H, W, 3) with values 0-255

    Returns:
        LAB image array (H, W, 3)
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

    # Scale to 0-255 range (OpenCV convention)
    L = L * 255 / 100
    a = a + 128
    b_ch = b_ch + 128

    lab = np.stack([L, a, b_ch], axis=2)
    return np.clip(lab, 0, 255).astype(np.uint8)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert LAB image to RGB color space.

    Args:
        lab: LAB image array (H, W, 3) with values 0-255 (OpenCV convention)

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


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur using scipy.

    Args:
        image: 2D array to blur
        sigma: Standard deviation of Gaussian kernel

    Returns:
        Blurred array
    """
    return ndimage.gaussian_filter(image.astype(np.float32), sigma=sigma)


def morphological_open(mask: np.ndarray, size: int = 5) -> np.ndarray:
    """Apply morphological opening (erosion then dilation)."""
    struct = np.ones((size, size))
    eroded = ndimage.binary_erosion(mask, structure=struct)
    opened = ndimage.binary_dilation(eroded, structure=struct)
    return opened.astype(np.float32)


def morphological_close(mask: np.ndarray, size: int = 5) -> np.ndarray:
    """Apply morphological closing (dilation then erosion)."""
    struct = np.ones((size, size))
    dilated = ndimage.binary_dilation(mask, structure=struct)
    closed = ndimage.binary_erosion(dilated, structure=struct)
    return closed.astype(np.float32)


class SkinToneTransfer:
    """Skin-tone specific color transfer (PIL/NumPy implementation).

    Args
    ---
        skin_blend_factor (float): Blending factor for skin regions (0.0-1.0).
        hair_region_blend_factor (float): Blending factor for the top head region (0.0-1.0).
        background_blend_factor (float): Blending factor for background (0.0-1.0).
        skin_ycrcb_lower (tuple): Lower bounds for skin detection in YCrCb (Y, Cr, Cb).
            Default (0, 133, 77) works for many skin tones.
        skin_ycrcb_upper (tuple): Upper bounds for skin detection in YCrCb (Y, Cr, Cb).
            Default (255, 173, 127) works for many skin tones.

    Note on YCrCb bounds:
        - Y (luminance): 0-255, usually keep full range
        - Cr (red chroma): typical skin range 133-173, lower for darker skin, higher for lighter
        - Cb (blue chroma): typical skin range 77-127

        Presets for different skin tones:
        - Light skin: (0, 140, 77) to (255, 180, 127)
        - Medium skin: (0, 133, 77) to (255, 173, 127)  [default]
        - Dark skin: (0, 120, 77) to (255, 160, 140)
        - Wide range: (0, 120, 70) to (255, 180, 140)
    """

    def __init__(self, skin_blend_factor=0.9, hair_region_blend_factor=0.5, background_blend_factor=0.3,
                 skin_ycrcb_lower=(0, 133, 77), skin_ycrcb_upper=(255, 173, 127)):
        self.skin_blend_factor = skin_blend_factor
        self.hair_region_blend_factor = hair_region_blend_factor
        self.background_blend_factor = background_blend_factor

        # Configurable skin detection bounds in YCrCb
        self.skin_ycrcb_lower = np.array(skin_ycrcb_lower, dtype=np.uint8)
        self.skin_ycrcb_upper = np.array(skin_ycrcb_upper, dtype=np.uint8)

        # Store color statistics from source skin pixels
        self.source_mean = None
        self.source_std = None

        # Store masks for visualization
        self.source_skin_mask = None
        self.target_skin_mask = None
        self.hair_region_mask = None

    def _create_skin_mask(self, image, expand_face=True):
        """Create a mask of skin regions using color-based segmentation.

        Note: This version does NOT use face detection (no OpenCV).
        It relies purely on YCrCb color thresholding.

        Args
        ---
            image (numpy.ndarray): Input RGB image.
            expand_face (bool): Ignored in this implementation (no face detection).

        Returns
        ---
            tuple: (skin_mask, hair_region_mask) where skin_mask is 0-1 float values.
        """
        # Convert to YCrCb color space
        image_ycrcb = rgb_to_ycrcb(image)

        # Create a binary mask for skin color using configurable bounds
        y_mask = (image_ycrcb[:, :, 0] >= self.skin_ycrcb_lower[0]) & \
                 (image_ycrcb[:, :, 0] <= self.skin_ycrcb_upper[0])
        cr_mask = (image_ycrcb[:, :, 1] >= self.skin_ycrcb_lower[1]) & \
                  (image_ycrcb[:, :, 1] <= self.skin_ycrcb_upper[1])
        cb_mask = (image_ycrcb[:, :, 2] >= self.skin_ycrcb_lower[2]) & \
                  (image_ycrcb[:, :, 2] <= self.skin_ycrcb_upper[2])

        skin_region = (y_mask & cr_mask & cb_mask).astype(np.float32)

        # Apply morphological operations to clean up the mask
        skin_region = morphological_open(skin_region, size=5)
        skin_region = morphological_close(skin_region, size=5)

        # Create empty hair region mask (no face detection available)
        # Approximate: top 20% of detected skin centroid region
        hair_region_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Find centroid of skin region to estimate head position
        if np.sum(skin_region) > 0:
            y_coords, x_coords = np.where(skin_region > 0.5)
            if len(y_coords) > 0:
                min_y = np.min(y_coords)
                # Hair region is above the top of detected skin
                hair_top = max(0, min_y - int(image.shape[0] * 0.1))
                hair_region_mask[hair_top:min_y, :] = 1.0

        # Normalize to 0-1 float
        skin_mask = skin_region.astype(np.float32)

        return skin_mask, hair_region_mask

    def _get_skin_statistics(self, image, skin_mask):
        """Calculate mean and standard deviation from skin pixels only.

        Args
        ---
            image (numpy.ndarray): RGB image.
            skin_mask (numpy.ndarray): Binary skin mask (0-1 values).

        Returns
        ---
            tuple: (mean, std) for LAB channels computed from skin pixels only.
        """
        # Convert to LAB
        image_lab = rgb_to_lab(image)

        # Create binary mask for indexing
        binary_mask = skin_mask > 0.5

        # Extract skin pixels
        skin_pixels_l = image_lab[:, :, 0][binary_mask]
        skin_pixels_a = image_lab[:, :, 1][binary_mask]
        skin_pixels_b = image_lab[:, :, 2][binary_mask]

        if len(skin_pixels_l) == 0:
            # Fallback to entire image if no skin detected
            print("Warning: No skin pixels detected in source, using entire image")
            skin_pixels_l = image_lab[:, :, 0].flatten()
            skin_pixels_a = image_lab[:, :, 1].flatten()
            skin_pixels_b = image_lab[:, :, 2].flatten()

        mean = np.array([np.mean(skin_pixels_l),
                        np.mean(skin_pixels_a),
                        np.mean(skin_pixels_b)])
        std = np.array([np.std(skin_pixels_l),
                       np.std(skin_pixels_a),
                       np.std(skin_pixels_b)])

        return mean, std

    def fit(self, image):
        """Learn the skin-tone statistics from the source image.

        Detects skin regions in the source image and computes LAB color
        statistics from only those pixels.

        Args
        ---
            image (numpy.ndarray): Source RGB image.
        """
        t0 = time.time()
        print(f"[TIMING] fit() started - image shape: {image.shape}")

        # Create skin mask for source image
        self.source_skin_mask, _ = self._create_skin_mask(image, expand_face=True)
        t0 = _log_time(t0, "fit: create_skin_mask")

        # Get statistics from skin pixels only
        self.source_mean, self.source_std = self._get_skin_statistics(
            image, self.source_skin_mask
        )
        t0 = _log_time(t0, "fit: get_skin_statistics")

        # Store source shape for reference
        self.source_shape = image.shape

        print(f"Source skin pixels: {np.sum(self.source_skin_mask > 0.5)}")
        print(f"Source skin LAB mean: L={self.source_mean[0]:.1f}, a={self.source_mean[1]:.1f}, b={self.source_mean[2]:.1f}")
        print(f"Source skin LAB std:  L={self.source_std[0]:.1f}, a={self.source_std[1]:.1f}, b={self.source_std[2]:.1f}")

    def recolor(self, image):
        """Apply skin-tone transfer to the target image.

        Args
        ---
            image (numpy.ndarray): Target RGB image.

        Returns
        ---
            numpy.ndarray: Color-transferred RGB image.
        """
        if self.source_mean is None or self.source_std is None:
            raise ValueError("You must call fit() before recolor()")

        t0 = time.time()
        print(f"[TIMING] recolor() started - image shape: {image.shape}")

        # Create skin mask for target image
        target_skin_mask_raw, self.hair_region_mask = self._create_skin_mask(image, expand_face=True)
        t0 = _log_time(t0, "recolor: create_skin_mask")

        # Apply Gaussian blur for smooth blending
        self.target_skin_mask = gaussian_blur(target_skin_mask_raw, sigma=5)
        t0 = _log_time(t0, "recolor: gaussian_blur skin_mask")

        # Get target skin statistics
        target_mean, target_std = self._get_skin_statistics(image, target_skin_mask_raw)
        t0 = _log_time(t0, "recolor: get_skin_statistics")

        print(f"Target skin pixels: {np.sum(target_skin_mask_raw > 0.5)}")
        print(f"Target skin LAB mean: L={target_mean[0]:.1f}, a={target_mean[1]:.1f}, b={target_mean[2]:.1f}")
        print(f"Target skin LAB std:  L={target_std[0]:.1f}, a={target_std[1]:.1f}, b={target_std[2]:.1f}")

        # Convert target to LAB
        target_lab = rgb_to_lab(image).astype(np.float32)
        t0 = _log_time(t0, "recolor: rgb_to_lab")

        # Apply Reinhard transfer to LAB channels
        result_lab = np.copy(target_lab)
        for i in range(3):
            if target_std[i] < 1e-8:
                continue
            result_lab[:, :, i] = ((target_lab[:, :, i] - target_mean[i]) *
                                   (self.source_std[i] / target_std[i]) +
                                   self.source_mean[i])
        t0 = _log_time(t0, "recolor: reinhard_transfer")

        # Clip and convert back to RGB
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        full_transfer = lab_to_rgb(result_lab)
        t0 = _log_time(t0, "recolor: lab_to_rgb")

        # Create background mask (areas that are not skin)
        background_mask = 1.0 - self.target_skin_mask

        # Apply blending based on masks
        result = np.zeros_like(image, dtype=np.float32)

        # Blend skin regions
        result += (self.target_skin_mask[:, :, np.newaxis] *
                  (self.skin_blend_factor * full_transfer +
                   (1 - self.skin_blend_factor) * image))
        t0 = _log_time(t0, "recolor: blend_skin")

        # Apply Gaussian blur to hair region mask
        if self.hair_region_mask is not None and np.sum(self.hair_region_mask) > 0:
            self.hair_region_mask = gaussian_blur(
                self.hair_region_mask.astype(np.float32), sigma=10
            )
        else:
            # Use top 25% of the image as approximation for hair region
            self.hair_region_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            hair_height = image.shape[0] // 4
            self.hair_region_mask[:hair_height, :] = 1.0
            self.hair_region_mask = gaussian_blur(self.hair_region_mask, sigma=10)
        t0 = _log_time(t0, "recolor: hair_region_mask")

        # Special handling for hair regions
        hair_region_exclusive = self.hair_region_mask * background_mask
        result += (hair_region_exclusive[:, :, np.newaxis] *
                  (self.hair_region_blend_factor * full_transfer +
                   (1 - self.hair_region_blend_factor) * image))
        t0 = _log_time(t0, "recolor: blend_hair")

        # Blend remaining background regions
        remaining_bg = background_mask * (1.0 - hair_region_exclusive)
        result += (remaining_bg[:, :, np.newaxis] *
                  (self.background_blend_factor * full_transfer +
                   (1 - self.background_blend_factor) * image))
        t0 = _log_time(t0, "recolor: blend_background")

        print("[TIMING] recolor() complete")
        return np.clip(result, 0, 255).astype(np.uint8)
