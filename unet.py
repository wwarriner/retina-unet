from tensorflow.python.keras.layers import (
    Activation,
    concatenate,
    Conv2D,
    Dropout,
    Input,
    MaxPooling2D,
    UpSampling2D,
)
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.losses import categorical_crossentropy, Loss, losses_utils
from tensorflow.python.keras.metrics import MeanIoU, Accuracy
from tensorflow.python.keras import backend as K
from functools import reduce


def compute_depth(base, level):
    return base * (2 ** level)


def build_convolution(previous, depth):
    SHAPE = (3, 3)
    ACT = "relu"
    PAD = "same"
    DROP = 0.2
    conv = Conv2D(depth, SHAPE, activation=ACT, padding=PAD)(previous)
    drop = Dropout(DROP)(conv)
    return Conv2D(depth, SHAPE, activation=ACT, padding=PAD)(drop)


def contract(previous):
    SHAPE = (2, 2)
    pool = MaxPooling2D(SHAPE)(previous)
    return pool


def expand(previous, transfer_conv):
    SHAPE = (2, 2)
    up = UpSampling2D(size=SHAPE)(previous)
    return concatenate([transfer_conv, up])


def activate(previous):
    DEPTH = 2
    SHAPE = (1, 1)
    ACT = "relu"
    PAD = "same"
    ACT = "softmax"
    act = Conv2D(DEPTH, SHAPE, activation=ACT, padding=PAD)(previous)
    return Activation(ACT)(act)


def downscale(start, depth, levels):
    convs = []
    pools = []
    previous = start
    for i in range(levels):
        current_depth = compute_depth(depth, i)
        conv = build_convolution(previous, current_depth)
        pool = contract(conv)
        convs.append(conv)
        pools.append(pool)
        previous = pool
    return convs, pools


def bottom_out(previous, depth, levels):
    bottom_depth = compute_depth(depth, levels)
    return build_convolution(previous, bottom_depth)


def upscale(bottom, down_convs, depth):
    convs = []
    ups = []
    previous = bottom
    levels = len(down_convs)
    for i in reversed(range(levels)):
        current_depth = compute_depth(depth, i)
        up = expand(previous, down_convs[i])
        conv = build_convolution(up, current_depth)
        ups.append(up)
        convs.append(conv)
        previous = conv
    return convs, ups


class WeightedCategoricalCrossentropy(Loss):
    def __init__(self, weight_vector):
        super().__init__(
            reduction=losses_utils.ReductionV2.AUTO,
            name="WeightedCategoricalCrossentropy",
        )
        self._weight_vector = weight_vector
        self._total_weight = sum(weight_vector)
        self._count = len(weight_vector)

    def call(self, y_true, y_pred):
        weights = [y_true[..., i] * self._weight_vector[i] for i in range(self._count)]
        weights = sum(weights) / self._total_weight
        return weights * categorical_crossentropy(y_true, y_pred)


def build_unet(input_shape, levels, learning_rate=0.01, weight_vector=None):
    CLASSES = 2
    if weight_vector is None:
        weight_vector = [1 / CLASSES for _ in range(CLASSES)]

    BASE_DEPTH = 32
    inputs = Input(input_shape)
    convs, pools = downscale(inputs, BASE_DEPTH, levels)
    bottom = bottom_out(pools[-1], BASE_DEPTH, levels)
    convs, _ = upscale(bottom, convs, BASE_DEPTH)
    act = activate(convs[-1])

    model = Model(inputs=inputs, outputs=act)
    model.compile(
        optimizer=SGD(lr=learning_rate),
        loss=WeightedCategoricalCrossentropy(weight_vector),
        metrics=[MeanIoU(num_classes=2), Accuracy()],
    )
    return model
