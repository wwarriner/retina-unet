from itertools import repeat
import json
from pathlib import PurePath
import random

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import config

devices = config.experimental.list_physical_devices("GPU")
assert len(devices) > 0
config.experimental.set_memory_growth(devices[0], True)
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python import one_hot
from tensorflow.python.keras.models import load_model

import image_preprocess as ip
from image_util import *
import patch_extract as pe
from unet import build_unet


def preprocess(images):
    # 1st dim is stack, 2nd & 3rd are spatial, 4th is channels
    out = np.array([ip.rgb2gray(image) for image in images])
    out = ip.standardize(out)
    out = ip.rescale(out, out_range=(0, 255)).astype(np.uint8)
    out = np.array([ip.apply_clahe(image) for image in out])
    out = np.array([ip.adjust_gamma(image, 1.2) for image in out])
    # functions stripped 4th dim, so add back
    out = out[..., np.newaxis]
    return out


def extract_random(images, patch_shape, patch_count, masks=None, auxiliary_images=None):
    # 1st dim is stack, 2nd & 3rd are spatial, 4th is channels
    assert images.ndim in (3, 4)
    if images.ndim == 4:
        assert images.shape[-1] in (1, 3)

    image_count = images.shape[0]
    patches_per_image = max(round(patch_count / image_count), 1)

    # generate_random_patch ignores the mask if none is provided
    # to use comprehensions with zip() we need an always-None iterable.
    if masks is None:
        masks = repeat(None)
    if auxiliary_images is None:
        auxiliary_images = repeat(None)
    raw = [
        pe.generate_random_patch(image, patch_shape, mask, aux)
        for _ in range(patches_per_image)
        for image, mask, aux in zip(images, masks, auxiliary_images)
    ]
    return map(np.stack, zip(*raw))


def load_config():
    with open("./config.json") as f:
        return json.load(f)


def save_architecture(json_str, json_file):
    with open(json_file, "w") as f:
        f.write(json_str)


def train():
    # tf.random.set_seed(352462476247)

    config = load_config()
    paths = config["paths"]
    training_folder = PurePath(".") / paths["data"] / paths["training"]

    training = stack(load_folder(str(training_folder / paths["images"]), ext=".tif"))
    training = preprocess(training) / 255
    masks = stack(load_folder(str(training_folder / paths["masks"]), ext=".gif"))
    groundtruth = (
        stack(load_folder(str(training_folder / paths["groundtruth"]), ext=".gif"))
        / 255
    ).astype(np.uint8)

    patch_shape = config["training"]["patch_shape"]
    patch_count = config["training"]["patch_count"]
    patches, groundtruth = extract_random(
        training, patch_shape, patch_count, masks, groundtruth
    )
    # visualize(montage(patches, (10, 10)), "patches")
    # visualize(montage(255 * groundtruth, (10, 10)), "groundtruth")

    unet_levels = config["training"]["unet_levels"]
    model = build_unet(patches.shape[1:], unet_levels)

    name = config["general"]["name"]
    out_folder = PurePath(".") / "out" / name
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    model_file = out_folder / (name + "_architecture.json")
    save_architecture(model.to_json(), str(model_file))

    checkpoint_file = out_folder / (name + "_best_weights.h5")
    checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_file),
        verbose=1,
        monitor="val_loss",
        mode="auto",
        save_best_only=True,
    )

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    split = config["training"]["validation_split"]
    model.fit(
        patches,
        one_hot(groundtruth.squeeze(), depth=2),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=split,
        callbacks=[checkpoint],
    )

    last_file = out_folder / (name + "_last_weights.h5")
    model.save_weights(str(last_file), overwrite=True)


def test():
    config = load_config()
    paths = config["paths"]
    testing_folder = PurePath(".") / paths["data"] / paths["training"]

    testing = stack(load_folder(str(testing_folder / paths["images"]), ext=".tif"))
    testing = preprocess(testing) / 255
    # masks = stack(load_folder(str(testing_folder / paths["masks"]), ext=".gif"))
    groundtruth = (
        stack(load_folder(str(testing_folder / paths["groundtruth"]), ext=".gif")) / 255
    ).astype(np.uint8)

    patch_shape = config["training"]["patch_shape"]
    patches, patch_counts, padding = patchify(testing, patch_shape)

    name = config["general"]["name"]
    out_folder = PurePath(".") / "out" / name
    model_file = out_folder / (name + "_best_weights.h5")
    model = load_model(model_file)

    predictions = model.predict(patches)
    predictions = np.argmax(predictions, axis=-1)
    predictions = predictions[..., np.newaxis]
    predictions = unpatchify(predictions, patch_counts, padding)
    # TODO mask out predictions
    # TODO write predicted images to files
    pass


# train()
test()
