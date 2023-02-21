import argparse
from PIL import Image
import numpy as np
import cv2
import os
import shutil
import tempfile
import string
import random


def build_argument_parser() -> dict:
    ap = argparse.ArgumentParser()
    image_group = ap.add_mutually_exclusive_group(required=True)
    image_group.add_argument("-s", "--source", help="path to a single input image")
    image_group.add_argument("-d", "--directory", help="path to directory of images")
    ap.add_argument("-t", "--target", required=True, help="path to reference image")
    ap.add_argument("-o", "--output", required=False, help="path to output directory")
    ap.add_argument("-c", "--color", required=False, type = int, help="color to transfer (e.g. red, green, blue, etc.)")
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


def get_image(filename: str):
    image= cv2.imread(filename)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.array(img)[:, :, :3]
    

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
    return cv2.flip(image, axis)


def split_image(image, image_name, tile_dim, output_dir, return_tile_dim=None):
    # convert image to BGR
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Get image dimensions
    rows, cols = tile_dim
    height, width, _ = img.shape

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

            # Extract the tile using cv2.rectangle
            tile = img[start_y:end_y, start_x:end_x]

            # Save the tile to the output directory with the filename format 'filename_{row}_{column}.jpg'
            filename = f'{image_name}_{row}_{col}.jpg'
            file_path = os.path.join(output_dir, filename)
            cv2.imwrite(file_path, tile)

            if return_tile_dim is not None and (row, col) == return_tile_dim:
                return file_path
      

def main():
    args = build_argument_parser()

    
if __name__ == "__main__":
    main()