# palette-transfer/helpers.py
import argparse
from PIL import Image
import numpy as np
import os
import shutil
import tempfile
import string
import random

def build_argument_parser() -> dict:
    ap = argparse.ArgumentParser(description="Image color palette transfer tool")
    
    # Image input arguments
    image_group = ap.add_mutually_exclusive_group(required=True)
    image_group.add_argument("-s", "--source", help="path to a single input image")
    image_group.add_argument("-d", "--directory", help="path to directory of images")
    ap.add_argument("-t", "--target", required=True, help="path to reference image")
    ap.add_argument("-o", "--output", required=False, help="path to output directory")
    
    # Color transfer method and parameters
    ap.add_argument("-m", "--method", required=False, default="all",
                    choices=["kmeans", "reinhard", "unique", "entire", "targeted", "skintone", "all"],
                    help="color transfer method to use: kmeans, reinhard, unique, entire, targeted, skintone, or all")
    ap.add_argument("-c", "--color", required=False, type=int, default=8,
                    help="number of colors in palette for k-means methods (default: 8)")
    ap.add_argument("--color-space", required=False, choices=["RGB", "LAB"], default="RGB",
                    help="color space for k-means clustering (default: RGB)")
    
    # Targeted transfer options
    ap.add_argument("--skin-blend", required=False, type=float, default=0.9,
                    help="blending factor for skin regions (0.0-1.0, default: 0.9)")
    ap.add_argument("--hair-blend", required=False, type=float, default=0.5,
                    help="blending factor for hair regions (0.0-1.0, default: 0.5)")
    ap.add_argument("--bg-blend", required=False, type=float, default=0.3,
                    help="blending factor for background (0.0-1.0, default: 0.3)")

    # Skin detection YCrCb bounds (for skintone method)
    ap.add_argument("--skin-cr-low", required=False, type=int, default=133,
                    help="lower Cr bound for skin detection (default: 133, use ~120 for darker skin)")
    ap.add_argument("--skin-cr-high", required=False, type=int, default=173,
                    help="upper Cr bound for skin detection (default: 173, use ~180 for lighter skin)")
    ap.add_argument("--skin-cb-low", required=False, type=int, default=77,
                    help="lower Cb bound for skin detection (default: 77)")
    ap.add_argument("--skin-cb-high", required=False, type=int, default=127,
                    help="upper Cb bound for skin detection (default: 127, use ~140 for darker skin)")

    # Additional options
    ap.add_argument("--random-walk", action="store_true", help="apply random walk effect to kmeans methods")
    ap.add_argument("--walk-steps", required=False, type=int, default=5,
                    help="number of steps for random walk effect (default: 5)")
    
    return vars(ap.parse_args())

def create_folder(folder_name: str):
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)


def copy_files_to_temp_folder(file1, file2):
    # get the directory where the Python script is run
    script_dir = os.path.dirname(os.path.abspath(__file__))

    while True:
        # generate a random folder name
        folder_name = ''.join(random.choices(string.ascii_lowercase, k=10))
        folder_path = os.path.join(script_dir, folder_name)
        if not os.path.exists(folder_path):
            # create the folder if it doesn't exist
            os.makedirs(folder_path)
            break

    # copy the files into the folder
    src_file1 = os.path.join(folder_path, f"src_{os.path.basename(file1)}")
    tgt_file2 = os.path.join(folder_path, f"tgt_{os.path.basename(file2)}")
    shutil.copy(file1, src_file1)
    shutil.copy(file2, tgt_file2)

    return folder_path, folder_name

def delete_folder_and_contents(folder_path: str):
    shutil.rmtree(folder_path)


def get_image(filepath: str, color_space: str = "BGR"):
    image = Image.open(filepath).convert('RGB')
    image_array = np.array(image)
    
    if color_space == "BGR":
        # Convert RGB to BGR
        return image_array[:, :, ::-1]
    elif color_space == "RGB":
        # Keep RGB color space
        return image_array
    elif color_space == "LAB":
        # Convert RGB to LAB using PIL
        lab_image = image.convert('LAB')
        return np.array(lab_image)
    elif color_space == "HSV":
        # Convert RGB to HSV using PIL
        hsv_image = image.convert('HSV')
        return np.array(hsv_image)
    else:
        # Raise an error if an invalid color space is provided
        raise ValueError("Invalid color space provided")    


def get_relevant_filepaths(directory, acceptable_formats):
    try:
        all_files = os.listdir(directory)
        if not all_files:
            raise Exception("No files found in directory: {}".format(directory))
        relevant_files = [f for f in all_files if any(f.endswith(format) for format in acceptable_formats)]
        if not relevant_files:
            raise Exception("No files with the specified formats found in directory: {}".format(directory))
    except Exception as e:
        print(e)

    file_paths = []
    for file in relevant_files:
        file_path = os.path.join(directory, file)
        file_paths.append(file_path)
    return file_paths


def get_unique_colors(filepath: str, color_space: str = "BGR"):
    # Load the image using get_image function
    image = get_image(filepath, color_space)

    # Convert the image to 1D array of colors
    colors = np.reshape(image, (-1, 3))

    # Get the unique colors and their counts
    unique_colors, counts = np.unique(colors, axis=0, return_counts=True)

    # Convert the unique colors to tuples and create a dictionary of counts
    color_counts = dict(zip(tuple(map(tuple, unique_colors)), counts))

    return color_counts


def get_image_stats(filepath: str, color_space: str = "BGR"):
    # Load the image using get_image function
    image = get_image(filepath, color_space)

    # Get the height and width of the image
    height, width = image.shape[:2]

    # Get the number of pixels in the image
    num_pixels = height * width

    # Get the number of unique colors in the image
    unique_colors = get_unique_colors(filepath, color_space)
    num_unique_colors = len(unique_colors)
    (ch1, ch2, ch3) = (image[:,:,0], image[:,:,1], image[:,:,2])
    (ch1_mean, ch1_std) = (np.mean(ch1), np.std(ch1))
    (ch2_mean, ch2_std) = (np.mean(ch2), np.std(ch2))
    (ch3_mean, ch3_std) = (np.mean(ch3), np.std(ch3))

    return height, width, num_pixels, num_unique_colors, ch1_mean, ch1_std, ch2_mean, ch2_std, ch3_mean, ch3_std


def closest_rect(n):
    '''Finds the closest height and width of a 2:1 rectangle given the number
    of pixels.

    Args
    ---
        n (int): the number of pixels in an image.
    '''
    k = 0
    while 2 * k ** 2 < n:
        k += 1
    
    return k, 2*k

def visualize_palette(palette, scale=0):
    '''Visualizes a palette as a rectangle of increasingly "bright" colors.

    This method first converts the RGB pixels into grayscale and sorts the
    grayscale pixel intensity as a proxy of sorting the RGB pixels. Then the
    pixels are reshaped into a 2:1 rectangle and displayed. If there are more
    fewer pixels tahn the size of the rectangle, the remaining pixels are given
    a generic gray color.

    Args
    ---
        palette (numpy.ndarray): the RGB pixels of a  color palette.
        scale (int): the scale factor to apply to the image of the palette.
    '''
    palette_gray = palette @ np.array([[0.21, 0.71, 0.07]]).T
    idx = palette_gray.flatten().argsort()
    h, w = closest_rect(palette.shape[0])
    palette_sorted = palette[idx]
    padding = (h*w) - palette.shape[0]
    
    if (h*w) > palette.shape[0]:
        palette_sorted = np.vstack(
            (palette_sorted, 51*np.ones((padding, 3), dtype=np.uint8))
        )

    palette_sorted = palette_sorted.reshape(h, w, 3)
    im = Image.fromarray(palette_sorted)
    
    if scale > 0:
        return im.resize((scale*im.width, scale*im.height), Image.NEAREST)

    return im

    
def invert_image(image, axis=0):
    if axis == 0:
        return np.flipud(image)
    elif axis == 1:
        return np.fliplr(image)
    else:
        return image


def split_image(image, image_name, tile_dim, output_dir, return_tile_dim=None):
    # Get image dimensions
    rows, cols = tile_dim
    height, width, _ = image.shape

    # Ensure return_tile_dim is valid
    if return_tile_dim is not None:
        row, col = return_tile_dim
        if row >= rows or col >= cols:
            raise ValueError("return_tile_dim must be less than tile_dim.")

    # Calculate tile width and height
    tile_width = width // cols
    tile_height = height // rows

    # Iterate through rows and cols
    for row in range(rows):
        for col in range(cols):
            if return_tile_dim is not None and (row, col) != return_tile_dim:
                continue
            # Calculate the starting and ending x and y coordinates for the tile
            start_x = col * tile_width
            end_x = (col + 1) * tile_width
            start_y = row * tile_height
            end_y = (row + 1) * tile_height

            # Extract the tile
            tile = image[start_y:end_y, start_x:end_x]

            # Save the tile to the output directory with the filename format 'filename_{row}_{column}.jpg'
            filename = f'{image_name}_{row}_{col}.jpg'
            file_path = os.path.join(output_dir, filename)
            tile_image = Image.fromarray(tile)
            tile_image.save(file_path)

            if return_tile_dim is not None and (row, col) == return_tile_dim:
                return file_path
      

def main():
    args = build_argument_parser()

    
if __name__ == "__main__":
    main()