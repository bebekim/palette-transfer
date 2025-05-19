# palette-transfer/plaette-transfer.py
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
from helpers import get_unique_colors, get_image_stats
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

        # Convert RGB to LAB
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

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
            
        # Convert back to RGB
        result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result


class TargetedReinhardTransfer:
    ''' Targeted Reinhard color transfer for medical before/after images.
    
    Uses OpenCV to detect face and perform skin segmentation, then applies the 
    Reinhard color transfer algorithm with different weights to skin areas.
    
    Args
    ---
        skin_blend_factor (float): Blending factor for skin regions (0.0-1.0).
        hair_region_blend_factor (float): Blending factor for the top head region (0.0-1.0).
        background_blend_factor (float): Blending factor for background (0.0-1.0).
    '''
    def __init__(self, skin_blend_factor=0.9, hair_region_blend_factor=0.5, background_blend_factor=0.3):
        self.skin_blend_factor = skin_blend_factor
        self.hair_region_blend_factor = hair_region_blend_factor
        self.background_blend_factor = background_blend_factor
        
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
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
        # Convert to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Create empty mask
        skin_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # If face detected
        face_detected = len(faces) > 0
        
        # Convert to YCrCb color space (good for skin detection)
        image_ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color bounds in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create a binary mask for skin color
        skin_region = cv2.inRange(image_ycrcb, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_region = cv2.morphologyEx(skin_region, cv2.MORPH_OPEN, kernel)
        skin_region = cv2.morphologyEx(skin_region, cv2.MORPH_CLOSE, kernel)
        
        # If we found a face, use it to refine the skin mask
        if face_detected:
            face_mask = np.zeros_like(skin_region)
            for (x, y, w, h) in faces:
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
                self.hair_region_mask = np.zeros_like(skin_region, dtype=np.float32)
                self.hair_region_mask[hair_y:y, x:x+w] = 1.0
            
            # Combine face detection with skin color detection
            skin_region = cv2.bitwise_and(skin_region, face_mask)
        
        # Normalize to 0-1 float
        skin_mask = skin_region.astype(np.float32) / 255.0
        
        # Apply Gaussian blur for smoother transitions
        skin_mask = cv2.GaussianBlur(skin_mask, (15, 15), 0)
        
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
            self.hair_region_mask = cv2.GaussianBlur(self.hair_region_mask, (31, 31), 0)
        
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


def main():
    args = build_argument_parser()
    method = args["method"]
    k_colors = args["color"]
    color_space = args["color_space"]
    random_walk = args.get("random_walk", False)
    walk_steps = args.get("walk_steps", 5)
    
    # Targeted transfer parameters
    skin_blend = args.get("skin_blend", 0.9)
    hair_blend = args.get("hair_blend", 0.5)
    bg_blend = args.get("bg_blend", 0.3)
    
    # Load source and target images
    src = get_image(args["source"], color_space="RGB")
    tgt = get_image(args["target"], color_space="RGB")

    # Create output directory if not specified
    output_dir = args["output"]
    if output_dir is None:
        # Create an output directory in the same folder as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Print information
    print(f"Source image: {args['source']} ({src.shape})")
    print(f"Target image: {args['target']} ({tgt.shape})")
    print(f"Output directory: {output_dir}")
    print(f"Color transfer method: {method}")
    
    # Save original images to output directory for comparison
    src_path = os.path.join(output_dir, f"source_{os.path.basename(args['source'])}")
    tgt_path = os.path.join(output_dir, f"target_{os.path.basename(args['target'])}")
    cv2.imwrite(src_path, cv2.cvtColor(src, cv2.COLOR_RGB2BGR))
    cv2.imwrite(tgt_path, cv2.cvtColor(tgt, cv2.COLOR_RGB2BGR))
    
    # [Keep existing code for other methods...]
    
    # Apply Targeted Reinhard color transfer
    if method in ["targeted", "all"]:
        print("Applying Targeted Reinhard color transfer...")
        print(f"Skin blend: {skin_blend}, Hair blend: {hair_blend}, Background blend: {bg_blend}")
        
        targeted = TargetedReinhardTransfer(
            skin_blend_factor=skin_blend,
            hair_region_blend_factor=hair_blend,
            background_blend_factor=bg_blend
        )
        targeted.fit(src)
        tgt_targeted = targeted.recolor(tgt)
        
        # Save result
        tgt_targeted_path = os.path.join(output_dir, 
            f"targeted_skin{skin_blend}_hair{hair_blend}_bg{bg_blend}_{os.path.basename(args['target'])}")
        cv2.imwrite(tgt_targeted_path, cv2.cvtColor(tgt_targeted, cv2.COLOR_RGB2BGR))
        print(f"Targeted recolored image saved to {tgt_targeted_path}")
        
        # Also save visualization of the masks
        if targeted.skin_mask is not None:
            # Create visualization of skin mask
            skin_vis = (targeted.skin_mask * 255).astype(np.uint8)
            skin_vis = cv2.applyColorMap(skin_vis, cv2.COLORMAP_JET)
            skin_mask_path = os.path.join(output_dir, f"skin_mask_{os.path.basename(args['target'])}")
            cv2.imwrite(skin_mask_path, skin_vis)
            
            # Create visualization of hair region mask
            if targeted.hair_region_mask is not None:
                hair_vis = (targeted.hair_region_mask * 255).astype(np.uint8)
                hair_vis = cv2.applyColorMap(hair_vis, cv2.COLORMAP_JET)
                hair_mask_path = os.path.join(output_dir, f"hair_region_mask_{os.path.basename(args['target'])}")
                cv2.imwrite(hair_mask_path, hair_vis)
            
            print(f"Mask visualizations saved to {output_dir}")
    
    print(f"All output saved to {output_dir}")


if __name__=="__main__":
    main()