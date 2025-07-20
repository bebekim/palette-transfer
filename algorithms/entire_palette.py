# ABOUTME: This file implements entire palette transfer using color distance calculations
# ABOUTME: Transfers all unique colors between images using optimized dask array processing

import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar


class EntirePalette():
    ''' Whole palette transferrer.

    Takes an image and store all the unique pixels and can perform transferring
    all the closests pixels between two images.

    Args
    ---
        chunk_size (int): number of chunks to divide an array into during
            computation.
    '''
    def __init__(self, chunk_size=1024):
        self.chunk_size = chunk_size
        self.source_pixels = None

    def _preprocess(self, image):
        ''' Preprocess an image.

        Check that an image has exactly 3 color channels (RGB) and that the
        data type is numpy.uint8 (i.e. values between 0 and 255). Then, convert
        the image from W x H x C into WH x C.

        Args
        ---
            image (numpy.ndarray): the image to be preprocessed.

        Returns
        ---
            numpy.ndarray: the flattened image keeping all RGB pixels in the
                columns.
        '''
        assert image.shape[-1] == 3, 'image must have exactly 3 color channels'
        assert image.dtype == 'uint8', 'image must be in np.uint8 type'

        # Flatten pixels, if not already.
        if len(image.shape) > 2:
            return image.reshape(-1, 3)

        return image

    def fit(self, image):
        ''' The fit function for the palette.
        
        Preprocesses the reference image, get the unique pixels in the image.

        Args
        ---
            image (numpy.ndarray): the reference image for this palette.
        '''
        self.source_pixels = np.unique(self._preprocess(image), axis=0)

    def recolor(self, image):
        ''' Transfers all colors from the reference image to the input image.

        1) Keep the shape of the input image and preprocess.
        2) Convert the unique pixels array into a dask array and similarly for
            the input image.
        3) Perform the distance calculation between every pair of pixels using
            the color distance.
        4) Map the closest pixels of the reference image to the input image.
        5) Recolor.

        Args
        ---
            image (numpy.ndarray): the input image to be recolored.
        
        Returns
        ---
            numpy.ndarray: the recolored image.
        '''
        # 1) Keep the shape of the input image and preprocess.
        image_shape = image.shape
        image = self._preprocess(image)
        #  2) Convert the unique pixels array into a dask array and similarly for the input image.
        image_colors = np.unique(image, axis=0)

        #  3) Perform the distance calculation between every pair of pixels using the color distance.
        self_da = da.from_array(
            self.source_pixels.astype(np.long), chunks=(self.chunk_size, 3)
        )
        other_da = da.from_array(
            image_colors.reshape(-1, 1, 3).astype(np.long),
            chunks=(self.chunk_size, 1, 3)
        )
        
        rmean_da = (other_da[:, :, 0] + self_da[:, 0]) // 2
        rgb_da = other_da - self_da
        r_da = ((512 + rmean_da) * rgb_da[:, :, 0] ** 2) >> 8
        g_da = 4 * rgb_da[:, :, 1] ** 2
        b_da = ((767 - rmean_da) * rgb_da[:, :, 2] ** 2) >> 8
        result_da = (r_da + g_da + b_da).argmin(axis=1)
        
        with ProgressBar():
            mapping_idx = result_da.compute()
        
        colormap = {tuple(a): tuple(b)
                    for a, b in 
                    zip(image_colors, self.source_pixels[mapping_idx])}
        
        image_recolored = np.array([colormap[tuple(rgb)] for rgb in image])
        return image_recolored.reshape(image_shape).astype(np.uint8)