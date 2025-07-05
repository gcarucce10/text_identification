import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Multiply, Activation
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, Dense
from tensorflow.keras.layers import Input, Activation, MaxPooling2D, Reshape

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
        # Loads charset
        with open(f'resources/bluche_bressay_charset.txt','r') as f:
            self.charset = f.readline()
        # NAO MUDAR
        # Loads model with custom layers and optimizer
        (input_data, output_data) = self.param_dict[param]["architecture"]()
        self.model     = Model(input_data, output_data)
        self.optimizer = NormalizedOptimizer(tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.1))
        self.model.compile(optimizer=self.optimizer, loss=ctc_loss_lambda_func)
        self.model.load_weights(self.param_dict[param]["weights"])
        # Loads model charset for output decoding
        #file = open(self.param_dict[param]["charset"], 'r')
        #self.charset = str(file.read())
        #file.close()

    def decode(self, text):
        decoded = "".join([self.charset[int(x)] for x in text if x > -1])
        decoded = decoded.replace("¶", "").replace("¤", "")
        return decoded

    def predict(self, image_list):
        # Get model output
        output = self.model.predict(image_list)

        # CTC decode
        batch_size = int(np.ceil(len(output) / 1))
        input_length = len(max(output, key=len))

        predicts, probabilities = [], []

        index = 0
        until = batch_size
        x_test = np.asarray(output[index:until])
        x_test_len = np.asarray([input_length for _ in range(len(x_test))])
        decode, log = K.ctc_decode(x_test,
                                   x_test_len,
                                   greedy=False,
                                   beam_width=10,
                                   top_paths=1)
        decode = np.swapaxes(decode, 0, 1)
        predicts.extend([[[int(p) for p in x if p != -1] for x in y] for y in decode])
        probabilities.extend([np.exp(x) for x in log])
        predicts = [[self.decode(x) for x in y] for y in predicts]
        return (predicts, probabilities)

    

# TODO: Testes de funcionalidade
if __name__=='__main__':
    def preproc(img_list):
        for index in range(len(img_list)):
            # Loads, if needed
            if isinstance(img_list[index], str):
                img_list[index] = cv2.imread(f'{img_list[index]}', cv2.IMREAD_GRAYSCALE)
            # Gets background info
            u, i = np.unique(np.array(img_list[index]).flatten(), return_inverse=True)
            background = int(u[np.argmax(np.bincount(i))])
            # Finds max scale factor for no distortion
            h, w = np.asarray(img_list[index]).shape
            #print(f'(preproc) Orig shape: h:{h}, w:{w}')
            wt, ht, _ = 1024, 128, 1
            f = max((w / wt), (h / ht))
            new_size = (max(min(wt, int(w/f)), 1), max(min(ht, int(h/f)), 1))
            # Resizes with no distortion (completes one dimension with background color)
            img_list[index] = cv2.resize(img_list[index], new_size)
            target = np.ones([ht, wt], dtype=np.uint8) * background
            target[0:new_size[1], 0:new_size[0]] = img_list[index]
            img_list[index] = cv2.transpose(target)
            # Normalizes image
            img_list[index] = img_list[index].astype(np.float32)
            mean  = np.mean(img_list[index])
            std   = np.std(img_list[index])
            img_list[index] = (img_list[index] - mean) / std
        # Treat for bluche model input
        img_list = np.asarray(img_list).astype(np.float32)
        return np.expand_dims(img_list, axis=-1)
    
    model    = HTRModel("bluche")
    model.model.summary()
    img      = preproc(['test/model/inputs/line1.png'])
    #print(f'After preproc: {img.shape}')
    #print(img[0])
    #cv2.imwrite('test/model/outputs/test_input.png', (img[0]*255).astype(np.uint8))
    predicts = model.predict(img)
    print(predicts[0])
    