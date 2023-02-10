import argparse
from PIL import Image
import numpy as np

def build_argument_parser() -> dict:
    ap = argparse.ArgumentParser()
    image_group = ap.add_mutually_exclusive_group(required=True)
    image_group.add_argument("-s", "--source", help="path to a single input image")
    image_group.add_argument("-d", "--directory", help="path to directory of images")
    ap.add_argument("-c", "--color", required=True, type = int, help="color to transfer (e.g. red, green, blue, etc.)")
    ap.add_argument("-t", "--target", required=True, help="path to reference image")
    ap.add_argument("-o", "--output", required=False, help="path to output directory")
    return vars(ap.parse_args())


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


if __name__ == "__main__":
    args = build_argument_parser()
