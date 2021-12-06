""" Functions that format images """
import cv2
import numpy as np


def remap_image(
    image: np.ndarray,
    flow: np.ndarray,
    border_mode: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """
    Takes an image and warps it forwards in time according to the flow field.

    Args:
        image: The grayscale image to warp.
        flow: A 3D array.  The first two dimensions must be the same size as the first two
            dimensions of the image.  The third dimension represented the x and y displacement.
        border_mode: One of cv2's BorderTypes such as cv2.BORDER_CONSTANT or cv2.BORDER_REPLICATE.
            If border_mode=cv2.BORDER_CONSTANT then the border will be set to -1.
            For details of other border_mode settings, see the Open CV docs here:
            docs.opencv.org/4.5.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5

    Returns:  Warped image.
    """
    # Adapted from https://github.com/opencv/opencv/issues/11068
    height, width = flow.shape[:2]
    remap = -flow.copy()
    remap[..., 0] += np.arange(width)  # map_x
    remap[..., 1] += np.arange(height)[:, np.newaxis]  # map_y
    # remap docs:
    # docs.opencv.org/4.5.4/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4
    # TODO: Maybe use integer remap: docs say that might be faster?
    remapped_image = cv2.remap(
        src=image,
        map1=remap,
        map2=None,
        interpolation=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=-1,
    )
    return remapped_image


def crop_center(image: np.ndarray, output_image_size_pixels: int) -> np.ndarray:
    """
    Crop center of a 2D numpy image.

    Args:
        image: The input image to crop.
        output_image_size_pixels: The requested size of the output image.

    Returns:
        The cropped image, of size output_image_size_pixels x output_image_size_pixels
    """
    input_size_y, input_size_x = image.shape
    assert (
        input_size_x >= output_image_size_pixels
    ), "output_image_size_pixels is larger than the input image!"
    assert (
        input_size_y >= output_image_size_pixels
    ), "output_image_size_pixels is larger than the input image!"
    half_output_image_size_pixels = output_image_size_pixels // 2
    start_x = (input_size_x // 2) - half_output_image_size_pixels
    start_y = (input_size_y // 2) - half_output_image_size_pixels
    end_x = start_x + output_image_size_pixels
    end_y = start_y + output_image_size_pixels
    return image[start_y:end_y, start_x:end_x]
