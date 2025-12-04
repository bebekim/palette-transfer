# -*- coding: utf-8 -*-
# algorithms/skintone_transfer.py
"""
Skin-tone specific color transfer.

Transfers skin-tone from the source image to the target image by:
1. Detecting skin pixels in the source image
2. Computing LAB color statistics from only those skin pixels
3. Applying Reinhard color transfer to the target's skin regions

This ensures that only skin-tone colors are used for the transfer,
ignoring background, hair, and other non-skin regions in the source.
"""

import numpy as np
import cv2


class SkinToneTransfer:
    """Skin-tone specific color transfer.

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

        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Store color statistics from source skin pixels
        self.source_mean = None
        self.source_std = None

        # Store masks for visualization
        self.source_skin_mask = None
        self.target_skin_mask = None
        self.hair_region_mask = None

    def _create_skin_mask(self, image, expand_face=True):
        """Create a mask of skin regions using color-based segmentation.

        Args
        ---
            image (numpy.ndarray): Input RGB image.
            expand_face (bool): Whether to expand face region for better coverage.

        Returns
        ---
            tuple: (skin_mask, hair_region_mask) where skin_mask is 0-1 float values.
        """
        # Convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detect faces
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # Create empty masks
        hair_region_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # If face detected
        face_detected = len(faces) > 0

        # Convert to YCrCb color space (good for skin detection)
        image_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)

        # Create a binary mask for skin color using configurable bounds
        skin_region = cv2.inRange(image_ycrcb, self.skin_ycrcb_lower, self.skin_ycrcb_upper)

        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_region = cv2.morphologyEx(skin_region, cv2.MORPH_OPEN, kernel)
        skin_region = cv2.morphologyEx(skin_region, cv2.MORPH_CLOSE, kernel)

        # If we found a face, use it to refine the skin mask
        if face_detected:
            face_mask = np.zeros_like(skin_region)
            for (x, y, w, h) in faces:
                if expand_face:
                    # Expand the face region slightly
                    ex = int(w * 0.1)  # Expand by 10% on each side
                    ey = int(h * 0.1)
                    x = max(0, x - ex)
                    y = max(0, y - ey)
                    w = min(image.shape[1] - x, w + 2 * ex)
                    h = min(image.shape[0] - y, h + 2 * ey)

                # Mark the face area in the mask
                face_mask[y:y+h, x:x+w] = 255

                # Mark the top of the head as potential hair region
                hair_y = max(0, y - h//2)
                hair_region_mask[hair_y:y, x:x+w] = 1.0

            # Combine face detection with skin color detection
            skin_region = cv2.bitwise_and(skin_region, face_mask)

        # Normalize to 0-1 float
        skin_mask = skin_region.astype(np.float32) / 255.0

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
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

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
        # Create skin mask for source image
        self.source_skin_mask, _ = self._create_skin_mask(image, expand_face=True)

        # Get statistics from skin pixels only
        self.source_mean, self.source_std = self._get_skin_statistics(
            image, self.source_skin_mask
        )

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

        # Create skin mask for target image
        target_skin_mask_raw, self.hair_region_mask = self._create_skin_mask(image, expand_face=True)

        # Apply Gaussian blur for smooth blending
        self.target_skin_mask = cv2.GaussianBlur(target_skin_mask_raw, (15, 15), 0)

        # Get target skin statistics
        target_mean, target_std = self._get_skin_statistics(image, target_skin_mask_raw)

        print(f"Target skin pixels: {np.sum(target_skin_mask_raw > 0.5)}")
        print(f"Target skin LAB mean: L={target_mean[0]:.1f}, a={target_mean[1]:.1f}, b={target_mean[2]:.1f}")
        print(f"Target skin LAB std:  L={target_std[0]:.1f}, a={target_std[1]:.1f}, b={target_std[2]:.1f}")

        # Convert target to LAB
        target_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Apply Reinhard transfer to LAB channels
        result_lab = np.copy(target_lab)
        for i in range(3):
            if target_std[i] < 1e-8:
                continue
            result_lab[:, :, i] = ((target_lab[:, :, i] - target_mean[i]) *
                                   (self.source_std[i] / target_std[i]) +
                                   self.source_mean[i])

        # Clip and convert back to RGB
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        full_transfer = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

        # Create background mask (areas that are not skin)
        background_mask = 1.0 - self.target_skin_mask

        # Apply blending based on masks
        result = np.zeros_like(image, dtype=np.float32)

        # Blend skin regions
        result += (self.target_skin_mask[:, :, np.newaxis] *
                  (self.skin_blend_factor * full_transfer +
                   (1 - self.skin_blend_factor) * image))

        # Apply Gaussian blur to hair region mask
        if self.hair_region_mask is not None:
            self.hair_region_mask = cv2.GaussianBlur(
                self.hair_region_mask.astype(np.float32), (31, 31), 0
            )
        else:
            # Use top 25% of the image as approximation for hair region
            self.hair_region_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            hair_height = image.shape[0] // 4
            self.hair_region_mask[:hair_height, :] = 1.0
            self.hair_region_mask = cv2.GaussianBlur(self.hair_region_mask, (31, 31), 0)

        # Special handling for hair regions
        hair_region_exclusive = self.hair_region_mask * background_mask
        result += (hair_region_exclusive[:, :, np.newaxis] *
                  (self.hair_region_blend_factor * full_transfer +
                   (1 - self.hair_region_blend_factor) * image))

        # Blend remaining background regions
        remaining_bg = background_mask * (1.0 - hair_region_exclusive)
        result += (remaining_bg[:, :, np.newaxis] *
                  (self.background_blend_factor * full_transfer +
                   (1 - self.background_blend_factor) * image))

        return np.clip(result, 0, 255).astype(np.uint8)
