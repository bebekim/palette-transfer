# ABOUTME: This file implements K-means based palette reduction for medical image processing
# ABOUTME: Provides basic and unique K-means clustering with random walk variations for hair clinic analysis

import numpy as np
from sklearn.cluster import KMeans


class KMeansReducedPalette:
    ''' The K-means reduced palette class.

    Takes an image and performs k-means on all the RGB pixels of the image. The
    value of k is equal to `num_colors`.

    Args
    ---
        num_colors (int): the number of colors in the reduced palette.
    '''
    def __init__(self, num_colors):
        self.num_colors = num_colors
        self.kmeans = KMeans(num_colors, random_state=0xfee1600d)
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
        
        Preprocesses the reference image and perform k-means clustering on the
        pixels. Then, find the distance of each pixel to the nearest centroid.

        Args
        ---
            image (numpy.ndarray): the reference image for this palette.
        '''
        image_cpy = image.copy()
        self.source_pixels = self._preprocess(image_cpy)
        self.kmeans.fit(self.source_pixels)
        
        self.centroid_nearest_pixels = []

        for ci in range(self.num_colors):
            pixels_ci = self.source_pixels[self.kmeans.labels_ == ci]
            distances_ci = np.sqrt(np.sum(np.square(
                pixels_ci - self.kmeans.cluster_centers_[ci]), axis=1))
            pixels_ci = pixels_ci[np.argsort(distances_ci)]

            self.centroid_nearest_pixels.append(pixels_ci)

    def recolor(self, image):
        ''' Transfer the reduced palette onto another image.

        Takes an image, applies k-means clustering on the pixels of the image,
        replace the predicted cluster center's color onto the image.

        Args
        ---
            image (numpy.ndarray): the input image for palette reduction.

        Returns
        ---
            numpy.ndarray: the recolored image based on this reduced palette.
        '''
        original_shape = image.shape
        image = self._preprocess(image)
        recolor_idx = self.kmeans.predict(image)
        recolor = self.kmeans.cluster_centers_[recolor_idx]
        recolor = np.round(recolor).astype(np.uint8)  # Round back to 0-255.

        return recolor.reshape(original_shape)
    
    def random_walk_recolor(self, image, max_steps):
        ''' Recolor with a random walk.

        Does recoloring and adds a random value depending on the `max_steps`.
        Any values outside the range of [0, 255] are clipped to keep the image
        "visible".

        Args
        ---
            image (numpy.ndarray): the input image for palette reduction.
            max_steps (int): maximum number of RGB value to move.

        Returns
        ---
            numpy.ndarray: the recolored image with random walk applied.
        '''
        original_shape = image.shape
        image = self._preprocess(image)
        centroid_idxs = self.kmeans.predict(image)
        start = np.round(self.kmeans.cluster_centers_[centroid_idxs])

        diff = np.zeros(image.shape)

        for _ in range(max_steps):
            walk = np.eye(3)[np.random.randint(0, 3, size=image.shape[0])]
            sign = np.random.choice([-1, 1], size=(image.shape[0], 1))
            diff += walk * sign

        recolor = np.clip(start + diff, 0, 255).astype(np.uint8)

        return recolor.reshape(original_shape)
    
    def random_neighborhood_walk_recolor(self, image, max_steps):
        ''' Recolor with a random walk.

        Does recoloring and adds randomly jumps to the closests neighbors. The
        maximum number of jumps is given by `max_steps`.

        Args
        ---
            image (numpy.ndarray): the input image for palette reduction.
            max_steps (int): maximum number of RGB value to move.

        Returns
        ---
            numpy.ndarray: the recolored image with random neighborhood walk
                applied.
        '''
        original_shape = image.shape
        image = self._preprocess(image)
        recolor = image.copy()
        centroid_idxs = self.kmeans.predict(image)

        for ci in range(self.num_colors):
            n_pixels = np.sum(centroid_idxs == ci)

            if n_pixels == 0:
                continue

            n_neighbors = self.centroid_nearest_pixels[ci].shape[0]
            # Don't jump to a neighbor further than the furthest neighbor.
            neighbor_idxs = np.random.randint(min(max_steps, n_neighbors),
                                              size=(n_pixels))
            recolor_ci = self.centroid_nearest_pixels[ci][neighbor_idxs]
            recolor[centroid_idxs == ci] = recolor_ci
        
        return recolor.reshape(original_shape)


class UniqueKMeansReducedPalette(KMeansReducedPalette):
    ''' The K-means reduced palette class.

    Takes an image and performs k-means on the _unique_ RGB pixels of the
    image. The value of k is equal to `num_colors`.

    Args
    ---
        num_colors (int): the number of colors in the reduced palette.
    '''
    def __init__(self, num_colors):
        super().__init__(num_colors)

    def fit(self, image):
        ''' The fit function for the palette.
        
        Preprocesses the reference image, get the unique pixels in the image
        and call the fit function of the KMeansReducedPalette class.

        Args
        ---
            image (numpy.ndarray): the reference image for this palette.
        '''
        image_cpy = image.copy()
        pixels = self._preprocess(image_cpy)
        super().fit(np.unique(pixels, axis=0))