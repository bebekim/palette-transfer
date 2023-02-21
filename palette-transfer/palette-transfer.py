import math
import os
# from PIL import Image
# from imageio import imread
from dask.diagnostics import ProgressBar
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
from helpers import build_argument_parser, get_image, copy_files_to_temp_folder, split_image
from helpers import visualize_palette

# Refer to https://www.youtube.com/watch?v=iU7cVd9LnFo for tutorial
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
        image_shape = image.shape
        image = self._preprocess(image)
        image_colors = np.unique(image, axis=0)

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


    
def main():
    args = build_argument_parser()
    k_colors = args["color"]
    src = get_image(args["source"], color_space="RGB")
    tgt = get_image(args["target"], color_space="RGB")

    temp_folder_path, temp_folder_name = copy_files_to_temp_folder(args["source"], args["target"])
    print(f"Files copied into {temp_folder_name}")
    print(f"Folder path: {temp_folder_path}")
    print(f"Splitting images into tiles...")

    # return_src_tile_path = split_image(src, image_name="src", tile_dim=(4, 6), output_dir=temp_folder_path, return_tile_dim=(0, 0))    
    return_src_tile_path = split_image(src, image_name="src", tile_dim=(4, 6), output_dir=temp_folder_path, return_tile_dim=(0, 0))
    return_tgt_tile_path = split_image(tgt, image_name="tgt", tile_dim=(4, 6), output_dir=temp_folder_path, return_tile_dim=(0, 0))
    # print a message indicating the folder name
    # print the return_src_tile_path
    print(f"Source tiles saved in {return_src_tile_path}")
    print(f"Target tiles saved in {return_tgt_tile_path}")
    
    # src_tile = get_image(return_src_tile_path)
    # tgt_tile = get_image(return_tgt_tile_path)
    
    # # palette reduction using k-means    
    # palette_reduced = KMeansReducedPalette(k_colors)
    # palette_reduced.fit(src_tile)
    # palette_reducecd_visualised = visualize_palette(np.round(palette_reduced.kmeans.cluster_centers_).astype(np.uint8), scale=32)
    # # source_folder = os.path.dirname(args["source"])
    # palette_reducecd_visualised.save(os.path.join(temp_folder_path, "palette_src_tile.png"))


    # palette_reduced = KMeansReducedPalette(k_colors)
    # palette_reduced.fit(tgt_tile)
    # palette_reducecd_visualised = visualize_palette(np.round(palette_reduced.kmeans.cluster_centers_).astype(np.uint8), scale=32)
    # # source_folder = os.path.dirname(args["source"])
    # palette_reducecd_visualised.save(os.path.join(temp_folder_path, "palette_tgt_tile.png"))


    # tgt_recolor = palette_reduced.recolor(tgt)
    # tgt_folder = os.path.dirname(args["target"])
    # file_path = os.path.join(tgt_folder, "tgt_recolor.png")
    # cv2.imwrite(file_path, cv2.cvtColor(tgt_recolor, cv2.COLOR_RGB2BGR))

    # delete the folder and its contents


if __name__=="__main__":
    main()