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


# TODO can't change name parameter of layers, have to set as we go...
# TODO expose hyperparameters
# TODO fix tests
class Unet:
    def __init__(self):
        self._level_count = 2
        self._base_filter_count = 32
        self._input_shape = (48, 48)
        self._convolution_activation = "relu"
        self._final_activation = "softmax"
        self._padding = "same"
        self._dropout_rate = 0.2
        self._convolution_shape = (3, 3)
        self._pooling_shape = (2, 2)
        self._learning_rate = 1
        self._loss = None

        self._level = 0

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        self._loss = value

    def build(self):
        assert self._loss is not None

        input_layer = self._build_input()
        input_layer.name = "INPUT"

        convs = []
        pools = []
        previous = input_layer
        for _ in range(self._level_count):
            conv, pool = self._build_contraction_block(previous)
            previous = pool
            convs.append(conv)
            pools.append(pool)

        bottom = self._build_convolution_block(previous, "BOTTOM")
        previous = bottom

        skips = []
        for conv in reversed(convs):
            skip = self._build_expansion_block(conv, previous)
            previous = skip
            skips.append(skip)

        activation = self._build_activation()(previous)

        model = Model(inputs=input_layer, outputs=activation)
        model.compile(
            optimizer=SGD(lr=self._learning_rate),
            loss=self._loss,
            metrics=[MeanIoU(num_classes=2), Accuracy()],
        )
        return model

    def _build_expansion_block(self, contraction_conv, previous_layer):
        PREFIX = "EXPANSION"
        conv = self._build_convolution_block(previous_layer, PREFIX)
        up_sample = self._build_up_sample()(conv)
        up_sample.name = self._build_name(PREFIX, "UP")
        skip = self._build_skip_connection(contraction_conv, up_sample)(up_sample)
        skip.name = self._build_name(PREFIX, "SKIP")
        self.level -= 1
        return skip

    def _build_contraction_block(self, previous_layer):
        PREFIX = "CONTRACTION"
        conv = self._build_convolution_block(previous_layer, PREFIX)
        pool = self._build_max_pool()(conv)
        pool.name = self._build_name(PREFIX, "PL")
        self._level += 1
        return pool, conv

    def _build_convolution_block(self, previous_layer, prefix):
        conv = self._build_convolution()(previous_layer)
        conv.name = self._build_name(prefix, "C1")
        dropout = self._build_dropout()(conv)
        dropout.name = self._build_name(prefix, "DO")
        conv2 = self._build_convolution()(dropout)
        conv2.name = self._build_name(prefix, "C2")
        return conv2

    def _build_input(self):
        return Input(self._input_shape)

    def _build_convolution(self):
        return Conv2D(
            filter=self._filter_count,
            kernel_size=self._convolution_shape,
            activation=self._convolution_activation,
            padding=self._padding,
        )

    def _build_dropout(self):
        return Dropout(self._dropout_rate)

    def _build_max_pool(self):
        return MaxPooling2D(self._pooling_shape)

    def _build_up_sample(self):
        return UpSampling2D(size=self._pooling_shape)

    def _build_skip_connection(self, conv, up_sample):
        return concatenate([conv, up_sample])

    def _build_activation(self):
        conv = self._build_convolution()
        conv.kernel_size = (1, 1)
        conv.name = "FINAL_CONVOLUTION"
        act = Activation(activation="softmax")(conv)
        act.name = "ACTIVATE"
        return act

    def _build_name(self, prefix, identifier):
        return "{prefix:s}_L{level:s}_{id:s}".format(
            prefix=prefix, level=self._level, id=identifier
        )

    @property
    def _filter_count(self):
        return self._base_filter_count * (2 ** self._level)
