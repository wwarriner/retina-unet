import unittest
from pathlib import PurePath, Path
from math import ceil

import numpy as np

from image_util import *


class Test(unittest.TestCase):
    def setUp(self):
        self.side_len = np.iinfo(np.uint8).max
        self.base_shape = (self.side_len, self.side_len)

        self.rgb = np.moveaxis(np.indices(self.base_shape), 0, -1).astype(np.uint8)
        self.rgb = np.concatenate(
            (self.rgb, np.zeros(self.base_shape + (1,)).astype(np.uint8)), axis=2
        )
        self.rgb_shape = self.rgb.shape

        self.fov_radius_ratio = 0.45
        self.fov_offset = (-35, 35)
        self.fov_radius = floor(self.side_len * self.fov_radius_ratio)

        self.mask = generate_circular_fov_mask(
            self.base_shape, self.fov_radius, self.fov_offset
        )
        self.mask_shape = self.mask.shape

        self.wait_time = 500

        # ! set to be relatively prime to side_len
        self.patch_shape = (12, 12)

    def tearDown(self):
        pass

    def read_image(self):
        return cv2.imread("test.jpg")

    def show(self, image, tag, is_opencv=True):
        visualize(image, tag, is_opencv)
        cv2.moveWindow(tag, 100, 100)
        cv2.waitKey(self.wait_time)
        cv2.destroyWindow(tag)

    def test_stack(self):
        n = 3
        s = stack(n * (self.rgb,))
        self.assertEqual(s.shape[0], n)
        self.assertEqual(s.shape[1:], self.rgb.shape[0:])
        self.assertIsInstance(s, np.ndarray)

    def test_visualize(self):
        self.show(
            self.rgb.astype(np.uint8),
            "test: visualize_bgr (red and green??)",
            is_opencv=True,
        )
        cv2.waitKey(self.wait_time)
        self.show(
            self.rgb.astype(np.uint8),
            "test: visualize_rgb (blue and green??)",
            is_opencv=False,
        )
        cv2.waitKey(self.wait_time)
        self.show(
            (self.mask * 255).astype(np.uint8), "test: visualize_gray (is circle?)"
        )
        cv2.waitKey(self.wait_time)
        self.show(self.rgb[..., 0], "test: visualize_gray (is gradient?)")
        cv2.waitKey(self.wait_time)
        self.show(self.read_image(), "test: visualize_gray (is beachscape?)")
        cv2.waitKey(self.wait_time)

    def test_patchify(self):
        counts = np.array(
            [ceil(x / y) for x, y in zip(self.rgb.shape, self.patch_shape)]
        )
        count = counts.prod()
        patches = patchify(self.rgb, self.patch_shape)
        self.assertEqual(patches.ndim, self.rgb.ndim + 1)
        self.assertEqual(patches.shape[0], count)
        self.assertEqual(patches.shape[1:3], self.patch_shape)
        self.assertEqual(patches.shape[3], self.rgb.shape[2])

    def test_montage(self):
        patches = patchify(self.rgb, self.patch_shape)
        count = patches.shape[0]
        montage_len = floor(count ** 0.5)
        montage_shape = (montage_len, montage_len)
        m = montage(patches, montage_shape)
        self.show(m, "test: unshuffled rgb patches")
        cv2.waitKey(self.wait_time)
        # permute for visibility
        m = montage(np.random.permutation(patches), montage_shape)
        self.show(m, "test: shuffled rgb patches")
        cv2.waitKey(self.wait_time)

    def test_save_load(self):
        try:
            path = PurePath("image_util_test_output.png")
            save(str(path), self.rgb.astype(np.uint8))
            self.show(load(str(path)), "test: save/load")
            cv2.waitKey(1200)
        finally:
            if Path(path).is_file():
                Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
