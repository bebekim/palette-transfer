# ABOUTME: This file implements targeted Reinhard color transfer for medical before/after images
# ABOUTME: Uses face detection and skin segmentation for precise medical image processing in hair clinics

import numpy as np
from PIL import Image
from skimage import color, morphology, filters
from scipy import ndimage
import face_recognition
from .reinhard_transfer import ReinhardColorTransfer


def validate_blend_parameters(skin_blend, hair_blend, bg_blend):
    """Ensure algorithm parameters are within medical validity ranges"""
    assert 0.0 <= skin_blend <= 1.0, "Skin blend must be between 0.0 and 1.0"
    assert 0.0 <= hair_blend <= 1.0, "Hair blend must be between 0.0 and 1.0" 
    assert 0.0 <= bg_blend <= 1.0, "Background blend must be between 0.0 and 1.0"


class TargetedReinhardTransfer:
    ''' Targeted Reinhard color transfer for medical before/after images.
    
    Uses face_recognition library to detect face and perform skin segmentation, then applies the 
    Reinhard color transfer algorithm with different weights to skin areas.
    
    Args
    ---
        skin_blend_factor (float): Blending factor for skin regions (0.0-1.0).
        hair_region_blend_factor (float): Blending factor for the top head region (0.0-1.0).
        background_blend_factor (float): Blending factor for background (0.0-1.0).
    '''
    def __init__(self, skin_blend_factor=0.9, hair_region_blend_factor=0.5, background_blend_factor=0.3):
        validate_blend_parameters(skin_blend_factor, hair_region_blend_factor, background_blend_factor)
        
        self.skin_blend_factor = skin_blend_factor
        self.hair_region_blend_factor = hair_region_blend_factor
        self.background_blend_factor = background_blend_factor
        
        # Initialize the standard Reinhard transfer
        self.reinhard = ReinhardColorTransfer()
        
        # Store masks for visualization
        self.skin_mask = None
        self.hair_region_mask = None
    
    def _create_skin_mask(self, image):
        ''' Create a mask of skin regions using color-based segmentation.
        
        Args
        ---
            image (numpy.ndarray): Input RGB image.
            
        Returns
        ---
            numpy.ndarray: Mask of skin regions (0-1 float values).
        '''
        # Detect face locations using face_recognition
        face_locations = face_recognition.face_locations(image)
        
        # Create empty mask
        skin_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # If face detected
        face_detected = len(face_locations) > 0
        
        # Convert to YCbCr color space (good for skin detection)
        # YCbCr is equivalent to YCrCb but with different channel order
        image_ycbcr = color.rgb2ycbcr(image)
        
        # Define skin color bounds in YCbCr (adjusted for skimage format)
        # Y: luminance, Cb: blue-difference, Cr: red-difference
        y_channel = image_ycbcr[:, :, 0]
        cb_channel = image_ycbcr[:, :, 1] 
        cr_channel = image_ycbcr[:, :, 2]
        
        # Skin detection in YCbCr space (converted ranges from OpenCV YCrCb)
        skin_region = ((cr_channel >= 133) & (cr_channel <= 173) & 
                      (cb_channel >= 77) & (cb_channel <= 127))
        
        # Apply morphological operations to clean up the mask
        disk_kernel = morphology.disk(2)
        skin_region = morphology.opening(skin_region, disk_kernel)
        skin_region = morphology.closing(skin_region, disk_kernel)
        
        # If we found a face, use it to refine the skin mask
        if face_detected:
            face_mask = np.zeros_like(skin_region)
            for (top, right, bottom, left) in face_locations:
                # Expand the face region slightly
                h, w = bottom - top, right - left
                ex = int(w * 0.1)  # Expand by 10% on each side
                ey = int(h * 0.1)
                
                left = max(0, left - ex)
                top = max(0, top - ey)
                right = min(image.shape[1], right + ex)
                bottom = min(image.shape[0], bottom + ey)
                
                # Mark the face area in the mask
                face_mask[top:bottom, left:right] = True
                
                # Mark the top of the head as potential hair region
                hair_top = max(0, top - h//2)
                self.hair_region_mask = np.zeros_like(skin_region, dtype=np.float32)
                self.hair_region_mask[hair_top:top, left:right] = 1.0
            
            # Combine face detection with skin color detection
            skin_region = skin_region & face_mask
        
        # Normalize to 0-1 float
        skin_mask = skin_region.astype(np.float32)
        
        # Apply Gaussian blur for smoother transitions
        skin_mask = filters.gaussian(skin_mask, sigma=7.5)
        
        self.skin_mask = skin_mask
        return skin_mask
    
    def fit(self, image):
        ''' Learn the color statistics from the source image.
        
        Args
        ---
            image (numpy.ndarray): Source RGB image.
        '''
        # Fit the standard Reinhard transfer with the entire image
        self.reinhard.fit(image)
    
    def recolor(self, image):
        ''' Apply targeted color transfer to the target image.
        
        Args
        ---
            image (numpy.ndarray): Target RGB image.
            
        Returns
        ---
            numpy.ndarray: Color-transferred RGB image.
        '''
        # Create skin mask for target image
        skin_mask = self._create_skin_mask(image)
        
        # Create hair region mask if not already created by skin detection
        if self.hair_region_mask is None:
            # Use top 25% of the image as a simple approximation for hair region
            self.hair_region_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            hair_height = image.shape[0] // 4
            self.hair_region_mask[:hair_height, :] = 1.0
            
            # Apply Gaussian blur for smoother transitions
            self.hair_region_mask = filters.gaussian(self.hair_region_mask, sigma=15.5)
        
        # Create background mask (areas that are not skin)
        background_mask = 1.0 - skin_mask
        
        # Apply standard Reinhard transfer to entire image
        full_transfer = self.reinhard.recolor(image)
        
        # Apply blending based on masks
        result = np.zeros_like(image, dtype=np.float32)
        
        # Blend skin regions
        result += (skin_mask[:, :, np.newaxis] * 
                  (self.skin_blend_factor * full_transfer + 
                   (1 - self.skin_blend_factor) * image))
        
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