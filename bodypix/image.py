import logging
from collections import namedtuple
from typing import Optional, Sequence

import numpy as np
import cv2
import PIL.Image

LOGGER = logging.getLogger(__name__)


ImageSize = namedtuple('ImageSize', ('height', 'width'))


def get_image_size(image: np.ndarray):
    height, width, *_ = image.shape
    return ImageSize(height=height, width=width)


def _resize_image_to(
    image_array: np.ndarray,
    image_size: ImageSize,
    resize_method: Optional[str] = None
) -> np.ndarray:
    assert not resize_method or resize_method == 'bilinear'
    if len(image_array.shape) == 4:
        assert image_array.shape[0] == 1
        image_array = image_array[0]

    if image_array.shape[-1] == 1:
        image_array = image_array[..., 0]
        resize_image_array = cv2.resize(image_array, dsize=image_size[::-1])
        resize_image_array = np.expand_dims(resize_image_array, -1)
    else:
        resize_image_array = cv2.resize(image_array, dsize=image_size[::-1])
    LOGGER.debug(
        'resize_image_array image: %r (%r)', image_array.shape, resize_image_array.dtype
    )
    return resize_image_array


def resize_image_to(
    image_array: np.ndarray,
    image_size: ImageSize,
    resize_method: Optional[str] = None
) -> np.ndarray:
    if get_image_size(image_array) == image_size:
        return image_array
    return _resize_image_to(image_array, image_size, resize_method)


def crop_and_resize_batch(  # pylint: disable=too-many-locals
    image_array_batch: np.ndarray,
    boxes: Sequence[Sequence[float]],
    box_indices: Sequence[int],
    crop_size: Sequence[int],
    method='bilinear',
) -> np.ndarray:
    assert list(box_indices) == [0]
    assert len(boxes) == 1
    assert len(crop_size) == 2
    box = np.array(boxes[0])
    assert np.min(box) >= 0
    assert np.max(box) <= 1
    y1, x1, y2, x2 = list(box)
    assert y1 <= y2
    assert x1 <= x2
    assert len(image_array_batch) == 1
    image_size = get_image_size(image_array_batch[0])
    image_y1 = int(y1 * (image_size.height - 1))
    image_y2 = int(y2 * (image_size.height - 1))
    image_x1 = int(x1 * (image_size.width - 1))
    image_x2 = int(x2 * (image_size.width - 1))
    cropped_image_array = image_array_batch[0][
        image_y1:(1 + image_y2), image_x1: (1 + image_x2), :
    ]
    resized_cropped_image_array = resize_image_to(
        cropped_image_array, ImageSize(height=crop_size[0], width=crop_size[1])
    )
    return np.expand_dims(resized_cropped_image_array, 0)


def load_image(
    local_image_path: str,
    image_size: Optional[ImageSize] = None,
    max_size = 672
) -> np.ndarray:
    with PIL.Image.open(local_image_path) as image:
        ratio = max(image.size[0]/max_size, image.size[1]/max_size)
        if ratio > 1:
            image = image.resize((int(image.size[0]/ratio), int(image.size[1]/ratio)))
        image_array = np.asarray(image)
    if image_size is not None:
        image_array = resize_image_to(image_array, image_size)
    return image_array
