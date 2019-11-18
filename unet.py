from tensorflow.python.keras.layers import (
    Activation,
    concatenate,
    Conv2D,
    Dropout,
    Input,
    MaxPooling2D,
    Permute,
    Reshape,
    UpSampling2D,
)
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.metrics import MeanIoU, Accuracy
from tensorflow.python.keras import backend as K


def compute_depth(base, level):
    return base * (2 ** level)


def build_convolution(previous, depth):
    SHAPE = (3, 3)
    ACT = "relu"
    PAD = "same"
    DROP = 0.2
    conv = Conv2D(depth, SHAPE, activation=ACT, padding=PAD)(previous)
    conv = Dropout(DROP)(conv)
    conv = Conv2D(depth, SHAPE, activation=ACT, padding=PAD)(conv)
    return conv


def contract(previous):
    SHAPE = (2, 2)
    pool = MaxPooling2D(SHAPE)(previous)
    return pool


def expand(previous, transfer_conv):
    SHAPE = (2, 2)
    up = UpSampling2D(size=SHAPE)(previous)
    up = concatenate([transfer_conv, up])
    return up


def activate(previous, pixels):
    DEPTH = 2
    SHAPE = (1, 1)
    ACT = "relu"
    PAD = "same"
    ACT = "sigmoid"
    act = Conv2D(DEPTH, SHAPE, activation=ACT, padding=PAD)(previous)
    # act = Reshape((DEPTH, pixels))(act)
    # act = Permute((2, 3, 1))(act)
    act = Activation(ACT)(act)
    return act


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


def build_unet(input_shape, levels):
    DEPTH = 32
    inputs = Input(shape=input_shape)
    down_convs, pools = downscale(inputs, DEPTH, levels)
    bottom = bottom_out(pools[-1], DEPTH, levels)
    up_convs, _ = upscale(bottom, down_convs, DEPTH)
    outputs = activate(up_convs[-1], input_shape[:-1])
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=SGD(lr=0.01),
        loss=CategoricalCrossentropy(from_logits=False),
        metrics=[MeanIoU(num_classes=2), Accuracy()],
    )
    return model


# model = build_unet((48, 48, 1), 2)
# print(model.summary())
# pass
