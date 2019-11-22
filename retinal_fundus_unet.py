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
from tensorflow.python.keras.utils import to_categorical

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


def save_architecture(json_str, config):
    model_file = get_out_folder(config) / (get_name(config) + "_architecture.json")
    with open(model_file, "w") as f:
        f.write(json_str)


def get_name(config):
    return config["general"]["name"]


def get_out_folder(config):
    out_folder = PurePath(".") / "out" / get_name(config)
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    return out_folder


def get_train_test_subfolder(config, train_test, sub_tag):
    paths = config["paths"]
    return PurePath(".") / paths["data"] / paths[train_test] / paths[sub_tag]


def load_mask(config, test_train):
    paths = config["paths"]
    mask_folder = get_train_test_subfolder(config, test_train, "masks")
    mask = load_folder(str(mask_folder))
    mask = stack(mask)
    return mask / 255


def load_xy(config, test_train):
    paths = config["paths"]
    x_folder = get_train_test_subfolder(config, test_train, "images")
    x = load_folder(str(x_folder))
    x = stack(x)
    x = preprocess(x) / 255
    assert (x != 0).any()

    y_folder = get_train_test_subfolder(config, test_train, "groundtruth")
    y = load_folder(str(y_folder))
    y = stack(y)
    y = y / 255
    y = y.astype(np.uint8)
    assert (y == 1).any()
    assert (y == 0).any()

    return x, y


def extract_random_patches(config, x_train, y_train, masks):
    patch_shape = config["training"]["patch_shape"]
    patch_count = config["training"]["patch_count"]
    return extract_random(x_train, patch_shape, patch_count, masks, y_train)


def extract_structured_patches(config, x_test):
    patch_shape = config["training"]["patch_shape"]
    return patchify(x_test, patch_shape)


def create_model_checkpoint(config):
    name = get_name(config)
    out_folder = get_out_folder(config)
    checkpoint_file = out_folder / (name + "_best_weights.h5")
    return ModelCheckpoint(
        filepath=str(checkpoint_file),
        verbose=1,
        monitor="val_loss",
        mode="auto",
        save_best_only=True,
    )


def create_model(config, x_train):
    unet_levels = config["training"]["unet_levels"]
    learning_rate = config["training"]["learning_rate"]
    return build_unet(x_train.shape[1:], unet_levels, learning_rate)


def fit_model(config, model, x_train, y_train, checkpoint):
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    split = config["training"]["validation_split"]
    model.fit(
        x_train,
        to_categorical(y_train.squeeze()),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=split,
        callbacks=[checkpoint],
    )


def save_last_weights(config, model):
    name = get_name(config)
    out_folder = get_out_folder(config)
    last_file = out_folder / (name + "_last_weights.h5")
    model.save_weights(str(last_file), overwrite=True)


def load_model_from_best_weights(config):
    name = get_name(config)
    out_folder = get_out_folder(config)
    model_file = out_folder / (name + "_best_weights.h5")
    return load_model(str(model_file))


def generate_predictions(model, x_train, masks, patch_counts, padding):
    predictions = model.predict(x_train)
    # predictions = np.argmax(predictions, axis=-1)
    predictions = predictions[..., 1] > 0.5
    predictions = predictions[..., np.newaxis]
    predictions = predictions.astype(np.float)
    predictions = unpatchify(predictions, patch_counts, padding)
    return mask_images(predictions, masks)


def save_predictions(config, predictions):
    name = PurePath(str(get_name(config)) + "_prediction.png")
    out_folder = get_out_folder(config)
    save_images(out_folder, name, predictions)


def train():
    # tf.random.set_seed(352462476247)
    config = load_config()

    TRAIN = "train"
    x_train, y_train = load_xy(config, TRAIN)
    mask = load_mask(config, TRAIN)
    x_train, y_train = extract_random_patches(config, x_train, y_train, mask)
    # visualize(montage(x_train, (10, 10)), "x_train sample")
    # visualize(montage(y_train, (10, 10)), "y_train sample")

    checkpoint = create_model_checkpoint(config)
    model = create_model(config, x_train)
    save_architecture(model.to_json(), config)

    fit_model(config, model, x_train, y_train, checkpoint)
    save_last_weights(config, model)


def test():
    config = load_config()

    TEST = "test"
    x_test, y_test = load_xy(config, TEST)
    mask = load_mask(config, TEST)
    x_test, patch_count, padding = extract_structured_patches(config, x_test)

    model = load_model_from_best_weights(config)

    predictions = generate_predictions(model, x_test, mask, patch_count, padding)
    save_predictions(config, predictions)


if __name__ == "__main__":
    # train()
    test()
