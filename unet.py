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

# TODO add tests


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


def build_unet(input_shape, levels, learning_rate):
    DEPTH = 32
    inputs = Input(input_shape)
    # convs, pools = downscale(inputs, DEPTH, levels)
    # bottom = bottom_out(pools[-1], DEPTH, levels)
    # convs, _ = upscale(bottom, convs, DEPTH)
    # act = activate(convs[-1])
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2, up1])
    conv4 = Conv2D(64, (3, 3), activation="relu", padding="same")(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1, up2])
    conv5 = Conv2D(32, (3, 3), activation="relu", padding="same")(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv5)
    #
    conv6 = Conv2D(2, (1, 1), activation="relu", padding="same")(conv5)
    ############
    act = Activation("softmax")(conv6)

    model = Model(inputs=inputs, outputs=act)
    # TODO figure out why metrics aren't working
    model.compile(
        optimizer=SGD(lr=learning_rate),
        loss=CategoricalCrossentropy(from_logits=False),
        metrics=[MeanIoU(num_classes=2), Accuracy()],
    )
    return model
