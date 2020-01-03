import unittest
import json
from pathlib import PurePath

import numpy as np
from tensorflow import config
from tensorflow.python.keras.utils import to_categorical

devices = config.experimental.list_physical_devices("GPU")
assert len(devices) > 0
config.experimental.set_memory_growth(devices[0], True)


from unet import *
from lib.python_image_utilities.image_util import stack


class Test(unittest.TestCase):
    def setUp(self):
        self.class_count = 2
        self.epochs = 3
        self.config_file = PurePath("test") / "unet_test_config.json"
        with open(self.config_file) as f:
            self.config = json.load(f)

    def tearDown(self):
        pass

    def generate_x(self, count):
        return stack([np.ones(self.config["input_shape"]) for _ in range(count)])

    def generate_y(self, count):
        return stack(
            [
                (np.random.normal(size=self.config["input_shape"]) > 0).astype(np.float)
                for _ in range(count)
            ]
        )

    def build_unet(self):
        unet = Unet(self.class_count, **self.config)
        unet.loss = WeightedCategoricalCrossentropy([1, 1])
        return unet.build()

    def test_stacking(self):
        count = 2
        x = self.generate_x(count)
        y = self.generate_y(count)
        model = self.build_unet()
        before = [(x.name, x.numpy()) for x in model.trainable_variables]
        model.fit(x, to_categorical(y.squeeze()))
        after = [(x.name, x.numpy()) for x in model.trainable_variables]
        for i, (b, a) in enumerate(zip(before, after)):
            self.assertTrue(
                (b[1] != a[1]).any(),
                "Layer {n} failed to train: {name}".format(n=i, name=b[0]),
            )

    def test_loss(self):
        count = 32
        x = self.generate_x(count)
        y = self.generate_y(count)
        y = to_categorical(y.squeeze())
        model = self.build_unet()
        self.assertGreater(model.loss(to_categorical(x.squeeze()), y), 0.0)

    def test_determinism(self):
        # TODO Not fully implemented
        # See https://pypi.org/project/tensorflow-determinism/
        count = 32 ** 2
        x = self.generate_y(count)
        y = x.copy()
        y = to_categorical(y.squeeze())
        model = self.build_unet()
        model.fit(x, y, epochs=self.epochs)
        p1 = model.predict(x)

        model2 = self.build_unet()
        model2.fit(x, y, epochs=self.epochs)
        p2 = model2.predict(x)

        # TODO assertEqual when determinism finalized past TF 2.1.0
        self.assertNotEqual(model.loss(p1, y), model2.loss(p2, y))

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
