import unittest

import numpy as np
import noise

from image_preprocess import *
from image_util import *


class Test(unittest.TestCase):
    def setUp(self):
        self.side_len = 400
        self.base_shape = (self.side_len, self.side_len)

    def generate_noise(self):
        octaves = 1
        scale = self.side_len * 0.1
        offsets = np.random.uniform(-1000 * self.side_len, 1000 * self.side_len, 2)
        X, Y = np.meshgrid(np.arange(self.base_shape[0]), np.arange(self.base_shape[1]))
        X = X + offsets[0]
        Y = Y + offsets[0]
        noise_maker = np.vectorize(
            lambda x, y: noise.pnoise2(x / scale, y / scale, octaves=octaves)
        )
        n = noise_maker(X, Y)
        n = ((n - n.min()) / (n.max() - n.min())) * 255
        return n.astype(np.uint8)

    def reduce_contrast(self, image):
        factor = 3.0
        minimum = 50
        return (np.round(image / factor) + 50).astype(np.uint8)

    def generate_image(self):
        image = self.generate_noise()
        return self.reduce_contrast(image)

    def read_image(self):
        image = cv2.imread("test.jpg")
        image = rgb2gray(image)
        return self.reduce_contrast(image)

    def run_fn(self, image, fn, *args, **kwargs):
        out = fn(image, *args, **kwargs)
        vis = np.concatenate((image, out), axis=0)
        tag = "test: {}".format(fn.__name__)
        visualize(vis, tag)
        cv2.moveWindow(tag, 100, 100)
        cv2.waitKey(1200)
        cv2.destroyWindow(tag)

    def standardize(self, image):
        standardized = standardize(image)
        return self.rescale(image)

    def rescale(self, image):
        return rescale(image, out_range=(0, 255)).astype(np.uint8)

    def test_apply_clahe(self):
        self.run_fn(self.read_image(), apply_clahe)
        self.run_fn(self.generate_image(), apply_clahe)

    def test_standardize(self):
        self.run_fn(self.read_image(), self.standardize)
        self.run_fn(self.generate_image(), self.standardize)

    def test_rescale(self):
        self.run_fn(self.read_image(), self.rescale)
        self.run_fn(self.generate_image(), self.rescale)

    def test_adjust_gamma(self):
        self.run_fn(self.read_image(), adjust_gamma, 2.0)
        self.run_fn(self.generate_image(), adjust_gamma, 2.0)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
