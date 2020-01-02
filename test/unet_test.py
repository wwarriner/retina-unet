import unittest
import os

seed_value = 314159
os.environ["PYTHONHASHSEED"] = str(seed_value)

import random

random.seed(seed_value)

import numpy as np

np.random.seed(seed_value)

import tensorflow as tf

tf.random.set_seed(seed_value)

from tensorflow import config

devices = config.experimental.list_physical_devices("GPU")
assert len(devices) > 0
config.experimental.set_memory_growth(devices[0], True)

from tensorflow.python import one_hot

from unet import *
from image_util import stack


class Test(unittest.TestCase):
    def setUp(self):
        self.base_shape = (48, 48, 1)
        self.levels = 2

    def tearDown(self):
        pass

    def generate_x(self, count):
        return stack([np.ones(self.base_shape) for _ in range(count)])

    def generate_y(self, count):
        return stack(
            [
                (np.random.normal(size=self.base_shape) > 0).astype(np.uint8)
                for _ in range(count)
            ]
        )

    def test_compute_depth(self):
        base = 32
        level = 2
        self.assertEqual(compute_depth(base, level), base * (2 ** level))

    def test_stacking(self):
        count = 2
        x = self.generate_x(count)
        y = self.generate_y(count)
        model = build_unet(self.base_shape, self.levels)
        before = [(x.name, x.numpy()) for x in model.trainable_variables]
        model.fit(x, one_hot(y.squeeze(), depth=2))
        after = [(x.name, x.numpy()) for x in model.trainable_variables]
        for i, (b, a) in enumerate(zip(before, after)):
            self.assertTrue(
                (b[1] != a[1]).any(),
                "Layer {n} failed to train: {name}".format(n=i, name=b[0]),
            )

    def test_loss(self):
        count = 32
        x = self.generate_y(count)
        x = one_hot(x.squeeze(), depth=2)
        y = self.generate_y(count)
        y = one_hot(y.squeeze(), depth=2)
        model = build_unet(self.base_shape, self.levels)
        self.assertGreater(model.loss(x, y), 0.0)

    def test_WeightedCategoricalCrossentropy(self):
        weights = (1.0, 9.0)
        wcce = WeightedCategoricalCrossentropy(weights)
        y_true = np.array([[[0, 1], [0, 1]], [[1, 0], [1, 0]]]).astype(np.float)
        y_pred = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]).astype(np.float)
        out = wcce(y_true, y_pred).numpy()
        self.assertGreater(out, 0.0)
        self.assertAlmostEqual(out, 4.03, 2)


if __name__ == "__main__":
    unittest.main()
