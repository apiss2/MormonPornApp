# based on:
# https://github.com/tensorflow/tfjs-models/blob/body-pix-v2.0.4/body-pix/src/util.ts

import math
from collections import namedtuple
from typing import Optional, Tuple, Union

import numpy as np

from .image import (
    crop_and_resize_batch,
    resize_image_to,
    ImageSize
)

class ResizeMethod:
    BILINEAR = 'bilinear'


Padding = namedtuple('Padding', ('top', 'bottom', 'left', 'right'))


# see isValidInputResolution
def is_valid_input_resolution(
    resolution: Union[int, float], output_stride: int
) -> bool:
    return (resolution - 1) % output_stride == 0


# see toValidInputResolution
def to_valid_input_resolution(
    input_resolution: Union[int, float], output_stride: int
) -> int:
    if is_valid_input_resolution(input_resolution, output_stride):
        return int(input_resolution)
    return int(math.floor(input_resolution / output_stride) * output_stride + 1)


# see toInputResolutionHeightAndWidth
def get_bodypix_input_resolution_height_and_width(
    internal_resolution_percentage: float,
    output_stride: int,
    input_height: int,
    input_width: int
) -> Tuple[int, int]:
    return (
        to_valid_input_resolution(
            input_height * internal_resolution_percentage, output_stride),
        to_valid_input_resolution(
            input_width * internal_resolution_percentage, output_stride)
    )


def _pad_image_like_tensorflow(
    image: np.ndarray,
    padding: Padding
) -> np.ndarray:
    """
    This is my padding function to replace with tf.image.pad_to_bounding_box
    :param image:
    :param padding:
    :return:
    """

    padded = np.copy(image)
    dims = padded.shape
    dtype = image.dtype

    if padding.top != 0:
        top_zero_row = np.zeros(shape=(padding.top, dims[1], dims[2]), dtype=dtype)
        padded = np.vstack([top_zero_row, padded])

    if padding.bottom != 0:
        bottom_zero_row = np.zeros(shape=(padding.top, dims[1], dims[2]), dtype=dtype)
        padded = np.vstack([padded, bottom_zero_row])

    dims = padded.shape
    if padding.left != 0:
        left_zero_column = np.zeros(shape=(dims[0], padding.left, dims[2]), dtype=dtype)
        padded = np.hstack([left_zero_column, padded])

    if padding.right != 0:
        right_zero_column = np.zeros(shape=(dims[0], padding.right, dims[2]), dtype=dtype)
        padded = np.hstack([padded, right_zero_column])

    return padded


# see padAndResizeTo
def pad_and_resize_to(
    image: np.ndarray,
    target_height, target_width: int
) -> Tuple[np.ndarray, Padding]:
    input_height, input_width = image.shape[:2]
    target_aspect = target_width / target_height
    aspect = input_width / input_height
    if aspect < target_aspect:
        # pads the width
        padding = Padding(
            top=0,
            bottom=0,
            left=round(0.5 * (target_aspect * input_height - input_width)),
            right=round(0.5 * (target_aspect * input_height - input_width))
        )
    else:
        # pads the height
        padding = Padding(
            top=round(0.5 * ((1.0 / target_aspect) * input_width - input_height)),
            bottom=round(0.5 * ((1.0 / target_aspect) * input_width - input_height)),
            left=0,
            right=0
        )

    padded = _pad_image_like_tensorflow(image, padding)
    resized = resize_image_to(
        padded, ImageSize(width=target_width, height=target_height)
    )
    return resized, padding


def get_images_batch(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 4:
        return image
    if len(image.shape) == 3:
        return np.expand_dims(image, axis=0)
    raise ValueError('invalid dimension, shape=%s' % str(image.shape))


# reverse of pad_and_resize_to
def remove_padding_and_resize_back(
    resized_and_padded: np.ndarray,
    original_height: int,
    original_width: int,
    padding: Padding,
    resize_method: Optional[str] = None
) -> np.ndarray:
    if not resize_method:
        resize_method = ResizeMethod.BILINEAR
    boxes = [[
        padding.top / (original_height + padding.top + padding.bottom - 1.0),
        padding.left / (original_width + padding.left + padding.right - 1.0),
        (
            (padding.top + original_height - 1.0)
            / (original_height + padding.top + padding.bottom - 1.0)
        ),
        (
            (padding.left + original_width - 1.0)
            / (original_width + padding.left + padding.right - 1.0)
        )
    ]]
    return crop_and_resize_batch(
        get_images_batch(resized_and_padded),
        boxes=boxes,
        box_indices=[0],
        crop_size=[original_height, original_width],
        method=resize_method
    )[0]


def get_sigmoid(x: np.ndarray):
    return 1/(1 + np.exp(-x))


# see scaleAndCropToInputTensorShape
def scale_and_crop_to_input_tensor_shape(
    image: np.ndarray,
    input_height: int,
    input_width: int,
    resized_height: int,
    resized_width: int,
    padding: Padding,
    apply_sigmoid_activation: bool = False,
    resize_method: Optional[str] = None
) -> np.ndarray:
    resized_and_padded = resize_image_to(
        image,
        ImageSize(height=resized_height, width=resized_width),
        resize_method=resize_method
    )
    if apply_sigmoid_activation:
        resized_and_padded = get_sigmoid(resized_and_padded)
    return remove_padding_and_resize_back(
        resized_and_padded,
        input_height, input_width,
        padding,
        resize_method=resize_method
    )
