# ABOUTME: This file implements the Reinhard color transfer algorithm for medical image standardization
# ABOUTME: Transforms color statistics between images using LAB color space for hair clinic documentation

import numpy as np
from PIL import Image
from skimage import color


class ReinhardColorTransfer:
    ''' The Reinhard color transfer class.

    Implements the color transfer method described in "Color Transfer between
    Images" by Erik Reinhard et al. (IEEE Computer Graphics and Applications, 2001).
    This method transforms the color statistics of the source image to match the
    target image in LAB color space.

    Args
    ---
        clip_values (bool): whether to clip pixel values to [0, 255] range
                           after transfer (default: True).
    '''
    def __init__(self, clip_values=True):
        self.clip_values = clip_values
        self.source_mean = None
        self.source_std = None

    def _preprocess(self, image):
        ''' Preprocess an image.

        Check that an image has exactly 3 color channels (RGB) and that the
        data type is numpy.uint8 (i.e. values between 0 and 255).

        Args
        ---
            image (numpy.ndarray): the image to be preprocessed in RGB format.

        Returns
        ---
            numpy.ndarray: the image converted to LAB color space.
        '''
        assert image.shape[-1] == 3, 'image must have exactly 3 color channels'
        assert image.dtype == 'uint8', 'image must be in np.uint8 type'

        # Convert RGB to LAB using skimage
        return (color.rgb2lab(image) * [100/100, 255/127, 255/127] + [0, 128, 128]).astype(np.uint8)

    def _get_mean_std(self, image):
        ''' Calculate mean and standard deviation for each channel

        Args
        ---
            image (numpy.ndarray): the image in LAB format.

        Returns
        ---
            tuple: (mean, std) where each is a numpy array with 3 values for L, a, b
        '''
        mean_l = np.mean(image[:, :, 0])
        mean_a = np.mean(image[:, :, 1])
        mean_b = np.mean(image[:, :, 2])
        
        std_l = np.std(image[:, :, 0])
        std_a = np.std(image[:, :, 1])
        std_b = np.std(image[:, :, 2])
        
        return (np.array([mean_l, mean_a, mean_b]), 
                np.array([std_l, std_a, std_b]))

    def fit(self, image):
        ''' The fit function for the color transfer.
        
        Preprocesses the source image and calculates mean and standard deviation
        for each LAB channel.

        Args
        ---
            image (numpy.ndarray): the source image (in RGB) for this color transfer.
        '''
        # Convert to LAB
        image_lab = self._preprocess(image.copy())
        
        # Calculate mean and std
        self.source_mean, self.source_std = self._get_mean_std(image_lab)
        
        # Store the original image shape
        self.source_shape = image.shape

    def recolor(self, image):
        ''' Transfers the color statistics from source to target.

        Takes a target image, converts it to LAB and applies the Reinhard
        color transfer algorithm to make its color statistics match the source.

        Args
        ---
            image (numpy.ndarray): the target image (in RGB) to be recolored.

        Returns
        ---
            numpy.ndarray: the recolored image (in RGB) based on the source's
                color statistics.
        '''
        if self.source_mean is None or self.source_std is None:
            raise ValueError("You must call fit() before recolor()")
            
        # Save target shape
        target_shape = image.shape
        
        # Convert target to LAB
        target_lab = self._preprocess(image.copy())
        
        # Calculate target statistics
        target_mean, target_std = self._get_mean_std(target_lab)
        
        # Create output image
        result = np.copy(target_lab).astype(np.float32)
        
        # Apply the color transfer
        for i in range(3):  # For each LAB channel
            # Skip if std is zero to avoid division by zero
            if target_std[i] < 1e-8:
                continue
                
            # Apply transfer function: ((x-μ_s)*(σ_t/σ_s))+μ_t
            result[:, :, i] = ((result[:, :, i] - target_mean[i]) * 
                              (self.source_std[i] / target_std[i]) + 
                              self.source_mean[i])
        
        # Clip values if needed
        if self.clip_values:
            result = np.clip(result, 0, 255)
            
        # Convert back to RGB using skimage
        lab_normalized = (result - [0, 128, 128]) / [100/100, 255/127, 255/127]
        result = (color.lab2rgb(lab_normalized) * 255).astype(np.uint8)
        
        return result