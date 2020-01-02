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
from tensorflow.keras.optimizers.schedules import (
    PiecewiseConstantDecay,
    ExponentialDecay,
)
from tensorflow.python.keras.losses import categorical_crossentropy, Loss, losses_utils
from tensorflow.python.keras.metrics import MeanIoU, Accuracy
from tensorflow.python.keras import backend as K
from functools import reduce


# TODO implement/test GIoU loss function


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


# TODO fix tests
class Unet:
    def __init__(self, class_count, **kwargs):

        assert isinstance(class_count, int)
        assert 1 < class_count
        self._class_count = class_count

        self._level_count = kwargs["level_count"]
        assert isinstance(self._level_count, int)
        assert 0 < self._level_count

        self._base_filter_count = kwargs["base_filter_count"]
        assert isinstance(self._base_filter_count, int)
        assert 2 ** (self._level_count + 1) <= self._base_filter_count

        self._input_shape = kwargs["input_shape"]
        assert isinstance(self._input_shape, (tuple, list))
        assert len(self._input_shape) == 3
        assert 0 < min(self._input_shape)

        self._convolution_activation = kwargs["convolution_activation"]
        assert self._convolution_activation in ("relu")

        self._final_activation = kwargs["final_activation"]
        assert self._final_activation in ("softmax")

        self._padding = kwargs["padding"]
        assert self._padding in ("same")

        self._dropout_rate = kwargs["dropout_rate"]
        assert isinstance(self._dropout_rate, float)
        assert 0.0 <= self._dropout_rate < 1.0

        self._kernel_size = kwargs["convolution_kernel_size"]
        assert isinstance(self._kernel_size, (tuple, list))
        assert len(self._kernel_size) == 2
        assert 0 < min(self._kernel_size)

        self._pooling_shape = kwargs["pooling_shape"]
        assert isinstance(self._pooling_shape, (tuple, list))
        assert len(self._pooling_shape) == 2
        assert 1 < min(self._pooling_shape)

        learning_rate = kwargs["learning_rate"]
        is_schedule = isinstance(learning_rate, dict)
        if not is_schedule:
            assert isinstance(learning_rate, (float, int))
            assert 0.0 < learning_rate < float("inf")
            learning_rate = lambda x: learning_rate
        elif "boundaries" in learning_rate:
            learning_rate = PiecewiseConstantDecay(**learning_rate)
        elif "decay_rate" in learning_rate:
            learning_rate = ExponentialDecay(**learning_rate)
        self._learning_rate = learning_rate

        self._loss = None
        self._level = 0

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        assert isinstance(value, Loss)
        self._loss = value

    def build(self):
        assert self._loss is not None
        self._level = 0

        input_layer = self._build_input()

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

        activation = self._build_activation(previous)

        model = Model(inputs=input_layer, outputs=activation)
        model.compile(
            optimizer=SGD(learning_rate=self._learning_rate),
            loss=self._loss,
            metrics=[MeanIoU(num_classes=2), Accuracy()],
        )
        return model

    def _build_expansion_block(self, contraction_conv, previous_layer):
        PREFIX = "EXPANSION"
        conv = self._build_convolution_block(previous_layer, PREFIX)
        name = self._build_name(PREFIX, "UP")
        up_sample = self._build_up_sample(name)(conv)
        name = self._build_name(PREFIX, "SKIP")
        skip = self._build_skip_connection(contraction_conv, up_sample, name)
        self._level -= 1
        return skip

    def _build_contraction_block(self, previous_layer):
        PREFIX = "CONTRACTION"
        conv = self._build_convolution_block(previous_layer, PREFIX)
        name = self._build_name(PREFIX, "PL")
        pool = self._build_max_pool(name)(conv)
        self._level += 1
        return conv, pool

    def _build_convolution_block(self, previous_layer, prefix):
        name = self._build_name(prefix, "C1")
        conv = self._build_convolution(name)(previous_layer)
        name = self._build_name(prefix, "DO")
        dropout = self._build_dropout(name)(conv)
        name = self._build_name(prefix, "C2")
        conv2 = self._build_convolution(name)(dropout)
        return conv2

    def _build_input(self):
        return Input(self._input_shape, name="INPUT")

    def _build_convolution(self, name):
        return Conv2D(
            filters=self._filter_count,
            kernel_size=self._kernel_size,
            activation=self._convolution_activation,
            padding=self._padding,
            name=name,
        )

    def _build_final_convolution(self):
        return Conv2D(
            filters=self._class_count,
            kernel_size=(1, 1),
            activation=self._convolution_activation,
            padding=self._padding,
            name="FINAL_CONVOLUTION",
        )

    def _build_dropout(self, name):
        return Dropout(self._dropout_rate, name=name)

    def _build_max_pool(self, name):
        return MaxPooling2D(self._pooling_shape, name=name)

    def _build_up_sample(self, name):
        return UpSampling2D(size=self._pooling_shape, name=name)

    def _build_skip_connection(self, conv, up_sample, name):
        return concatenate([conv, up_sample], name=name)

    def _build_activation(self, previous_layer):
        conv = self._build_final_convolution()(previous_layer)
        name = "ACTIVATE"
        act = Activation(activation=self._final_activation, name=name)(conv)
        return act

    def _build_name(self, prefix, identifier):
        return "{prefix:s}_L{level:s}_{id:s}".format(
            prefix=prefix, level=str(self._level), id=identifier
        )

    @property
    def _filter_count(self):
        return self._base_filter_count * (2 ** self._level)
