import os
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Multiply, Activation
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape

# Implementation from "Gated convolutional recurrent neural networks for multilingual handwriting recognition".
class GatedConv2D(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, **kwargs):
        super(GatedConv2D, self).__init__(**kwargs)

    def call(self, inputs):
        """Apply gated convolution"""

        output = super(GatedConv2D, self).call(inputs)
        linear = Activation("linear")(inputs)
        sigmoid = Activation("sigmoid")(output)

        return Multiply()([linear, sigmoid])

    def get_config(self):
        """Return the config of the layer"""

        config = super(GatedConv2D, self).get_config()
        return config

class NormalizedOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, name='normalized_optimizer', **kwargs):
        super().__init__(name, **kwargs)
        self.optimizer = optimizer
        self._learning_rate = optimizer.learning_rate

    def get_config(self):
        config = super().get_config()
        config.update({'optimizer': tf.keras.optimizers.serialize(self.optimizer)})
        return config

    def apply_gradients(self, grads_and_vars, name=None, skip_gradients_aggregation=False):

        if not skip_gradients_aggregation:
            grads_and_vars = [(grad / (tf.sqrt(tf.reduce_sum(tf.square(grad))) + 1e-7), var)
                              for grad, var in grads_and_vars if grad is not None]

        return self.optimizer.apply_gradients(grads_and_vars, name=name)

    @classmethod
    def from_config(cls, config):
        optimizer = tf.keras.optimizers.deserialize(config.pop('optimizer'))
        return cls(optimizer, **config)

@staticmethod
def ctc_loss_lambda_func(y_true, y_pred):
    """Function for computing the CTC loss"""

    y_true = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
    y_pred = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]))

    label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype=tf.int32)
    logit_length = tf.reduce_sum(tf.reduce_sum(y_pred, axis=-1), axis=-1, keepdims=True)

    ctc_loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, logit_length, label_length)
    ctc_loss = tf.reduce_mean(ctc_loss)

    return ctc_loss

def bluche(input_size=(1024, 128, 1), d_model=101):
    """
    Gated Convolucional Recurrent Neural Network by Bluche et al.

    Reference:
        Bluche, T., Messina, R.:
        Gated convolutional recurrent neural networks for multilingual handwriting recognition.
        In: Document Analysis and Recognition (ICDAR), 2017
        14th IAPR International Conference on, vol. 1, pp. 646–651, 2017.
        URL: https://ieeexplore.ieee.org/document/8270042
    """

    input_data = Input(name="input", shape=input_size)
    cnn = Reshape((input_size[0] // 2, input_size[1] // 2, input_size[2] * 4))(input_data)

    cnn = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="tanh")(cnn)

    cnn = Conv2D(filters=16, kernel_size=(2, 4), strides=(2, 4), padding="same", activation="tanh")(cnn)
    cnn = GatedConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="tanh")(cnn)
    cnn = GatedConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)

    cnn = Conv2D(filters=64, kernel_size=(2, 4), strides=(1, 4), padding="same", activation="tanh")(cnn)
    cnn = GatedConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same")(cnn)

    cnn = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="tanh")(cnn)
    cnn = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding="valid")(cnn)

    # This was originally "shape = cnn.get_shape()""
    shape = cnn.get_shape()
    blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    blstm = Dense(units=128, activation="tanh")(blstm)

    blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
    output_data = Dense(units=d_model, activation="softmax")(blstm)

    # NAO TIRA OS PARENTESES
    return (input_data, output_data)


class HTRModel:
    param_dict = {
        "bluche": dict(architecture=bluche,weights="resources/bluche_bressay_weights.hdf5",charset="resources/bluche_bressay_charset.txt")
    }

    def __init__(self, param = "bluche"):
        # NAO MUDAR
        # Loads model with custom layers and optimizer
        (input_data, output_data) = self.param_dict[param]["architecture"]()
        self.model     = Model(input_data, output_data)
        self.optimizer = NormalizedOptimizer(tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.1))
        self.model.compile(optimizer=self.optimizer, loss=ctc_loss_lambda_func)
        self.model.load_weights(self.param_dict[param]["weights"])
        # Loads model charset for output decoding
        file = open(self.param_dict[param]["charset"], 'r')
        self.charset = str(file.read())
        file.close()


# TODO: Testes de funcionalidade
if __name__=='__main__':
    model = HTRModel("bluche")
    print(f'MODEL\n')
    model.model.summary()
    print(f'\ncharset:\n{model.charset}')