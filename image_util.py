from math import floor
from pathlib import Path, PurePath
from itertools import cycle

import numpy as np
import cv2
from PIL import Image

from files import create_ext_glob


def get_center(shape):
    """Returns the center point of an image shape, rounded down."""
    return [floor(x / 2) for x in shape]


def generate_circular_fov_mask(shape, fov_radius, offset=(0, 0)):
    """Generates a circular field of view mask, with the interior of the circle
    included. The circle is assumed to be centered on the image shape."""
    center = get_center(shape)
    X, Y = np.meshgrid(np.arange(shape[0]) - center[0], np.arange(shape[1]) - center[1])
    R2 = (X - offset[0]) ** 2 + (Y - offset[1]) ** 2
    fov2 = fov_radius ** 2
    mask = R2 <= fov2
    return mask[..., np.newaxis]


def load_folder(path, ext):
    glob = create_ext_glob(ext)
    image_files = list(Path(path).glob(glob))
    return [load(str(image_file)) for image_file in image_files]


def stack(images):
    if type(images) is np.ndarray:
        images = (images,)
    images = [image[..., np.newaxis] if image.ndim == 2 else image for image in images]
    return np.stack(images)


def montage(images, shape):
    it = cycle(images)
    size = np.array(shape).prod()
    montage = np.array([next(it) for _ in range(size)])
    montage = montage.reshape(*shape, *images.shape[1:])
    montage = montage.transpose((0, 2, 1, 3))
    return montage.reshape([s * i for s, i in zip(shape, images.shape[1:])])


def save(path, image):
    cv2.imwrite(path, image)


def load(path):
    if PurePath(path).suffix in (".gif"):
        image = Image.open(path)
        image = np.asarray(image)
    else:
        image = cv2.imread(path)
    return image


def visualize(image, tag="UNLABELED_WINDOW", is_opencv=True):
    assert image.ndim in (2, 3)
    if image.ndim == 3:
        assert image.shape[-1] in (1, 3)

    cv2.namedWindow(tag, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(tag, image.shape[0:2][::-1])
    if is_opencv and image.ndim == 3 and image.shape[-1] == 3:
        cv2.imshow(tag, np.flip(image, axis=2))
    else:
        cv2.imshow(tag, image)
    cv2.waitKey(1)
