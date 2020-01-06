from itertools import repeat
from pathlib import Path, PurePath

import numpy as np
from tensorflow import config

devices = config.experimental.list_physical_devices("GPU")
assert len(devices) > 0
config.experimental.set_memory_growth(devices[0], True)
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.utils import to_categorical

from lib.config_snake.config import ConfigFile
from lib.python_image_utilities.image_preprocess import *
from lib.python_image_utilities.image_util import *
from lib.python_image_utilities.patch_extract import *
from unet import WeightedCategoricalCrossentropy, Unet

# TODO refactor functions here with functionality in other modules


def compute_class_count(y_train):
    y_categorical = to_categorical(y_train.squeeze())
    return y_categorical.shape[-1]


def compute_weights(y_train):
    y_categorical = to_categorical(y_train.squeeze())
    return [
        y_categorical[..., i].size / y_categorical[..., i].sum()
        for i in range(y_categorical.shape[-1])
    ]


def create_model(config, x_train, y_train):
    unet = Unet(compute_class_count(y_train), **config.training.unet.to_json())
    weight_vector = compute_weights(y_train)
    loss = WeightedCategoricalCrossentropy(weight_vector)
    unet.loss = loss
    return unet.build()


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


def extract_one_patch(config, x_train, y_train, masks):
    patch_shape = config.training.unet.input_shape
    patch_count = config.training.patch_count
    x, y = extract_random(x_train, patch_shape, 1, masks, y_train)
    x = np.concatenate([x for _ in range(patch_count)], axis=0)
    y = np.concatenate([y for _ in range(patch_count)], axis=0)
    return x, y


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
        generate_random_patch(image, patch_shape, mask, aux)
        for _ in range(patches_per_image)
        for image, mask, aux in zip(images, masks, auxiliary_images)
    ]
    return map(np.stack, zip(*raw))


def extract_random_patches(config, x_train, y_train, masks):
    patch_shape = config.training.unet.input_shape
    patch_count = config.training.patch_count
    return extract_random(x_train, patch_shape, patch_count, masks, y_train)


def extract_structured_patches(config, x_test):
    patch_shape = config.training.unet.input_shape[0:-1]
    return patchify(x_test, patch_shape)


def fit_model(config, model, x_train, y_train):
    epochs = config.training.epochs
    batch_size = config.training.batch_size
    split = config.training.validation_split
    checkpoint = create_model_checkpoint(config)
    model.fit(
        x_train,
        to_categorical(y_train.squeeze()),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=split,
        callbacks=[checkpoint],
    )


def generate_predictions(model, x_train, masks, patch_counts, padding):
    predictions = model.predict(x_train)
    predictions = predictions[..., 1] > 0.5
    predictions = predictions[..., np.newaxis]
    predictions = predictions.astype(np.float)
    predictions = unpatchify(predictions, patch_counts, padding)
    return (mask_images(predictions, masks) * 255).astype(np.uint8)


def get_name(config):
    return config.general.name


def get_out_folder(config):
    out_folder = PurePath(".") / config.paths.out / get_name(config)
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    return out_folder


def get_train_test_subfolder(config, train_test, sub_tag):
    paths = config.paths
    return PurePath(".") / paths.data / paths[train_test] / paths[sub_tag]


def load_mask(config, test_train):
    mask_folder = get_train_test_subfolder(config, test_train, "masks")
    mask = load_images(str(mask_folder), ".gif")
    mask = stack(mask)
    return mask / 255


# TODO fix duplication of filename suffixes
def load_model_from_best_weights(config):
    name = get_name(config)
    out_folder = get_out_folder(config)
    json_file = out_folder / (name + "_architecture.json")
    with open(json_file) as f:
        json_data = f.read()
    model = model_from_json(json_data)
    weights_file = out_folder / (name + "_best_weights.h5")
    model.load_weights(str(weights_file))
    return model


def load_xy(config, test_train):
    x_folder = get_train_test_subfolder(config, test_train, "images")
    x = load_images(str(x_folder), ".tif")
    x = stack(x)
    x = preprocess(x)
    assert (x != 0).any()

    y_folder = get_train_test_subfolder(config, test_train, "groundtruth")
    y = load_images(str(y_folder), ".gif")
    y = stack(y)
    y = y / 255
    assert (y == 1).any()
    assert (y == 0).any()

    return x, y


def preprocess(images):
    # 1st dim is stack, 2nd & 3rd are spatial, 4th is channels
    out = np.array([rgb2gray(image) for image in images])
    out = rescale(out, out_range=(0, 255)).astype(np.uint8)
    out = np.array([apply_clahe(image) for image in out])
    out = np.array([adjust_gamma(image, 1.2) for image in out])
    out = standardize(out)
    # functions stripped 4th dim, so add back
    out = out[..., np.newaxis]
    return out


def save_architecture(json_str, config):
    model_file = get_out_folder(config) / (get_name(config) + "_architecture.json")
    with open(model_file, "w") as f:
        f.write(json_str)


def save_last_weights(config, model):
    name = get_name(config)
    out_folder = get_out_folder(config)
    last_file = out_folder / (name + "_last_weights.h5")
    model.save_weights(str(last_file), overwrite=True)


def save_predictions(config, predictions):
    name = PurePath(str(get_name(config)) + "_prediction.png")
    out_folder = get_out_folder(config)
    save_images(out_folder, name, predictions)


def test(config=None):
    if config is None:
        config = ConfigFile("config.json")

    TEST = "test"
    x_test, y_test = load_xy(config, TEST)
    mask = load_mask(config, TEST)
    x_test, patch_count, padding = extract_structured_patches(config, x_test)

    model = load_model_from_best_weights(config)

    predictions = generate_predictions(model, x_test, mask, patch_count, padding)
    save_predictions(config, predictions)


def train(config=None):
    if config is None:
        config = ConfigFile("config.json")

    TRAIN = "train"
    x_train, y_train = load_xy(config, TRAIN)
    mask = load_mask(config, TRAIN)
    if config.debugging.enabled and config.debugging.single_training_example:
        x_train, y_train = extract_one_patch(config, x_train, y_train, mask)
    else:
        x_train, y_train = extract_random_patches(config, x_train, y_train, mask)

    if config.debugging.enabled and config.debugging.show_montages:
        visualize(montage(x_train, (10, 10)), "x_train sample")
        visualize(montage(x_train, (10, 10)), "x_train std sample")
        visualize(montage(y_train, (10, 10)), "y_train sample")

    model = create_model(config, x_train, y_train)
    save_architecture(model.to_json(), config)

    fit_model(config, model, x_train, y_train)
    save_last_weights(config, model)


def display_predictions(view_fn, config=None):
    if config is None:
        config = ConfigFile("config.json")

    predicted_images = stack(load_images(get_out_folder(config), ".png"))
    # predicted_images = np.stack([rgb2gray(image) for image in predicted_images])
    # predicted_images = predicted_images[..., np.newaxis]
    ground_truth_images = stack(
        load_images(get_train_test_subfolder(config, "test", "groundtruth"), ".gif")
    )
    overlay_montage = overlay(
        montage(predicted_images),
        montage(ground_truth_images).squeeze(),
        [0.0, 1.0, 0.0],
        0.5,
        0.5,
    )
    return view_fn(overlay_montage)


if __name__ == "__main__":
    display_predictions(visualize)
