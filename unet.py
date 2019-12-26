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

# TODO turn this into a class


def compute_depth(base, level):
    return base * (2 ** level)


def build_convolution(previous, depth, name):
    SHAPE = (3, 3)
    ACT = "relu"
    PAD = "same"
    DROP = 0.2
    conv = Conv2D(depth, SHAPE, activation=ACT, padding=PAD, name=name.format(id="C1"))(
        previous
    )
    drop = Dropout(DROP, name=name.format(id="DO"))(conv)
    return Conv2D(depth, SHAPE, activation=ACT, padding=PAD, name=name.format(id="C2"))(
        drop
    )


def contract(previous, depth, name):
    SHAPE = (2, 2)
    pool = MaxPooling2D(SHAPE, name=name)(previous)
    return pool


def expand(previous, transfer_conv, name, level):
    SHAPE = (2, 2)
    up = UpSampling2D(size=SHAPE, name=name)(previous)
    return concatenate([transfer_conv, up], name="CONCATENATE_L{:d}".format(level))


def activate(previous):
    DEPTH = 2
    SHAPE = (1, 1)
    ACT = "relu"
    PAD = "same"
    act = Conv2D(DEPTH, SHAPE, activation=ACT, padding=PAD, name="FINAL_CONV")(previous)
    return Activation("softmax", name="ACTIVATE")(act)


def downscale(start, depth, levels):
    convs = []
    pools = []
    previous = start
    for i in range(levels):
        current_depth = compute_depth(depth, i)
        name = "{name:s}_L{level:d}".format(name="CONTRACT", level=i) + "_{id:s}"
        conv = build_convolution(previous, current_depth, name)
        name = "POOL_FROM_{:d}_TO_{:d}".format(i, i + 1)
        pool = contract(conv, current_depth, name)
        convs.append(conv)
        pools.append(pool)
        previous = pool
    return convs, pools


def bottom_out(previous, depth, level):
    bottom_depth = compute_depth(depth, level)
    name = "{name:s}_L{level:d}".format(name="BOTTOM", level=level) + "_{id:s}"
    return build_convolution(previous, bottom_depth, name)


def upscale(bottom, down_convs, depth):
    convs = []
    ups = []
    previous = bottom
    levels = len(down_convs)
    for i in reversed(range(levels)):
        current_depth = compute_depth(depth, i)
        name = "UPSCALE_FROM_{:d}_TO_{:d}".format(i + 1, i)
        up = expand(previous, down_convs[i], name, i)
        name = "{name:s}_L{level:d}".format(name="EXPAND", level=i) + "_{id:s}"
        conv = build_convolution(up, current_depth, name)
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
