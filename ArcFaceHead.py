from typing import Optional

import tensorflow as tf

from tensorflow.keras import layers, regularizers
from tensorflow.keras import losses
from tensorflow.python.util import nest

from autokeras import adapters
from autokeras import analysers
from autokeras import hyper_preprocessors as hpps_module
from autokeras import preprocessors
from autokeras.blocks import reduction
from autokeras.engine import head as head_module
from autokeras.utils import types
from autokeras.utils import utils


from tensorflow.keras.layers import Layer

from tensorflow.keras import backend as K


class ArcFace(Layer):
    def __init__(self, n_classes=9691, s=16.0, m=0.20, regularizer=None, weight_decay=1e-4,
                 **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.weight_decay = weight_decay
        if regularizer is not None:
            self.regularizer = regularizers.get(regularizer)
        else:
            self.regularizer = regularizers.l2(weight_decay)

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                shape=(input_shape[0][-1], self.n_classes),
                                initializer='glorot_uniform',
                                trainable=True,
                                regularizer=self.regularizer)


    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_classes": self.n_classes,
                "s": self.s,
                "m": self.m,
                "regularizer": self.regularizer,
                "weight_decay": self.weight_decay,
            }
        )
        return config

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]

        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out


class ArcFaceHead(head_module.Head):
    """ArcFace Classification layers.
    Use sigmoid and binary crossentropy for binary classification and multi-label
    classification. Use softmax and categorical crossentropy for multi-class
    (more than 2) classification. Use Accuracy as metrics by default.
    The targets passing to the head would have to be tf.data.Dataset, np.ndarray,
    pd.DataFrame or pd.Series. It can be raw labels, one-hot encoded if more than two
    classes, or binary encoded for binary classification.
    The raw labels will be encoded to one column if two classes were found,
    or one-hot encoded if more than two classes were found.
    # Arguments
        num_classes: Int. Defaults to None. If None, it will be inferred from the
            data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use `binary_crossentropy` or
            `categorical_crossentropy` based on the number of classes [Recommended].
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        dropout: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        multi_label: bool = False,
        loss: Optional[types.LossType] = None,
        metrics: Optional[types.MetricsType] = None,
        dropout: Optional[float] = None,
        emb_dims: Optional[int] = None,
        face_weight_decay: Optional[float] = None,
        face_s: Optional[float] = None,
        face_m: Optional[float] = None,
        **kwargs
    ):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.dropout = dropout
        self.emb_dims = emb_dims
        if metrics is None:
            metrics = ["accuracy"]
        if loss is None:
            loss = self.infer_loss()
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        # Infered from analyser.
        self._encoded = None
        self._encoded_for_sigmoid = None
        self._encoded_for_softmax = None
        self._add_one_dimension = False
        self._labels = None
        self.face_s = face_s
        self.face_m = face_m
        self.face_weight_decay = face_weight_decay

    def infer_loss(self):
        if not self.num_classes:
            return None
        return losses.CategoricalCrossentropy()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "multi_label": self.multi_label,
                "dropout": self.dropout,
                "face_s": self.face_s,
                "face_m": self.face_m,
                "face_weight_decay": self.face_weight_decay,
            }
        )
        return config

    def build(self, hp, inputs):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 2)
        input_node = inputs[0]
        input_node2 = inputs[1]
        output_node = input_node

        # Reduce the tensor to a vector.
        if len(output_node.shape) > 2:
            output_node = reduction.SpatialReduction().build(hp, output_node)

        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0)

        if self.emb_dims is not None:
            emb_size = self.emb_dims
        else:
            emb_size = hp.Choice("emb_dims", [512, 1024], default=512)

        if dropout > 0:
            output_node = layers.Dropout(dropout)(output_node)

        output_node = layers.Dense(emb_size, name="arcface_embedding",  kernel_initializer='he_normal')(output_node)
        output_node = layers.BatchNormalization()(output_node)

        if self.face_s is not None:
            face_s = self.face_s
        else:
            if self.num_classes > 3000:
                face_s = hp.Choice("face_s", [8.0, 10.0, 12.5, 15.0, 17.5], default=10.0)
            elif 1000 < self.num_classes < 3000:
                face_s = hp.Choice("face_s", [10.0, 12.5, 15.0, 17.5], default=12.5)
            elif 1000 > self.num_classes > 200:
                face_s = hp.Choice("face_s", [12.5, 15.0, 17.5, 20.0], default=15.0)
            else:
                face_s = hp.Choice("face_s", [15.0, 17.5, 20.0, 25.0], default=17.5)


        if self.face_m is not None:
            face_m = self.face_m
        else:
            face_m = hp.Choice("face_m", [0.20, 0.35, 0.50], default=0.20)

        if self.face_weight_decay is not None:
            face_weight_decay = self.face_weight_decay
        else:
            face_weight_decay = hp.Choice("face_weight_decay", [1e-3, 1e-4, 1e-5], default=1e-4)

        output_node = ArcFace(name="arc_face", n_classes=self.num_classes, s=face_s, m=face_m, weight_decay=face_weight_decay)([output_node, input_node2])

        return output_node

    def get_adapter(self):
        return adapters.ClassificationAdapter(name=self.name)

    def get_analyser(self):
        return analysers.ClassificationAnalyser(
            name=self.name, multi_label=self.multi_label
        )

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self.num_classes = analyser.num_classes
        self.loss = self.infer_loss()
        self._encoded = analyser.encoded
        self._encoded_for_sigmoid = analyser.encoded_for_sigmoid
        self._encoded_for_softmax = analyser.encoded_for_softmax
        self._add_one_dimension = len(analyser.shape) == 1
        self._labels = analyser.labels

    def get_hyper_preprocessors(self):
        hyper_preprocessors = []

        if self._add_one_dimension:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.AddOneDimension())
            )

        if self.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.CastToInt32())
            )

        if not self._encoded and self.dtype != tf.string:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.CastToString())
            )

        if self._encoded_for_sigmoid:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.SigmoidPostprocessor()
                )
            )
        elif self._encoded_for_softmax:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.SoftmaxPostprocessor()
                )
            )
        elif self.num_classes == 2:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.LabelEncoder(self._labels)
                )
            )
        else:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.OneHotEncoder(self._labels)
                )
            )
        return hyper_preprocessors





