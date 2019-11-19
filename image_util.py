from math import floor
from pathlib import Path, PurePath
from itertools import cycle

import numpy as np
import cv2
from PIL import Image

from files import create_ext_glob

# TODO make this, image_preprocess.py, and patch_extract.py into a more generic
# TODO "image_stack" class


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
    hwc_shape = images.shape[1:]
    montage = montage.reshape(*shape, *hwc_shape)
    n = list(range(montage.ndim))
    montage = montage.transpose((0, 2, 1, 3, 4))
    hw_shape = hwc_shape[:-1]
    final_shape = [s * i for s, i in zip(shape, hw_shape)]
    final_shape.append(-1)
    return montage.reshape(final_shape)


def save(path, image):
    cv2.imwrite(path, image)


def load(path):
    if PurePath(path).suffix in (".gif"):
        image = Image.open(path)
        image = np.asarray(image)
    else:
        image = cv2.imread(path)
    return image


def patchify(images, patch_shape, *args, **kwargs):
    """image has HWC
    patch_shape has HW"""
    assert images.ndim in (3, 4)
    if images.ndim == 3:
        images = images[np.newaxis, ...]
    out_padding = patch_shape - np.remainder(images.shape[1:-1], patch_shape)
    padding = np.append(out_padding, 0)
    padding = np.insert(padding, 0, 0)
    padding = list(zip((0, 0, 0, 0), padding))
    padded = np.pad(images, padding, *args, **kwargs)
    # stack up patches
    patches = np.concatenate(
        [
            # images to patches - HWC to NHWC
            np.stack(
                [
                    padded[image, x : x + patch_shape[0], y : y + patch_shape[1], ...]
                    for x in range(0, padded.shape[1], patch_shape[0])
                    for y in range(0, padded.shape[2], patch_shape[1])
                ]
            )
            for image in range(padded.shape[0])
        ]
    )
    patch_counts = [x // y for x, y in zip(padded.shape[1:3], patch_shape)]
    return patches, patch_counts, list(out_padding)


def unpatchify(patches, patch_counts, padding):
    """patches has NHWC
    patch_counts has HW"""
    chunk_len = np.array(patch_counts).prod()
    # stack - N
    images = np.stack(
        [
            # rows - H
            np.concatenate(
                [
                    # cols - W
                    np.concatenate(
                        patches[y + chunk : y + patch_counts[0] + chunk], axis=0
                    )
                    for y in range(0, chunk_len, patch_counts[0])
                ],
                axis=1,
            )
            for chunk in range(0, patches.shape[0], chunk_len)
        ]
    )
    images = images[:, : -padding[0], : -padding[1], ...]
    return images


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
    # HACK Next three lines force the window to the top.
    # cv2.setWindowProperty(tag, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.waitKey(1)
    # cv2.setWindowProperty(tag, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.waitKey(1)
