import os
from abc import ABC
from xml.parsers.expat import model

import numpy as np
import cv2

from metrics import calculate_cer_wer

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Multiply, Activation
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, Dense
from tensorflow.keras.layers import Input, Activation, MaxPooling2D, Reshape

# Custom layers / tools
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

# TODO: get source
class NormalizedOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, name='normalized_optimizer', **kwargs):
        # 1. FIX: Define lr_variable by getting the learning_rate from the wrapped optimizer
        lr_variable = optimizer.learning_rate

        # O argumento learning_rate deve ser manipulado apenas pelo otimizador interno.
        super().__init__(name=name, **kwargs)

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


# Abstract class for generic HTR NN model
class HTRModel(ABC):
    # NEEDS IMPLEMENTING
    # Responsible for loading the model into self.model (kerasmodel) and charset into self.charset (str)
    # After __init__, self.model.predict(x) should already be working
    def __init__(self):
        pass

    # READY AS IS
    # Recieves text in model's encoding and decode it according to self.charset (removing pad. and unk. tokens)
    def decode(self, text, pad="¶", unk="¤"):
        decoded = "".join([self.charset[int(x)] for x in text if x > -1])
        decoded = decoded.replace(pad, "").replace(unk, "")
        return decoded

    # NEEDS IMPLEMENTING
    # Calls self.predict. By convention, recieves list and returns list of outputs
    def predict(self, image_list):
        pass

# Class for custom bluche model trained on bressay
class Bluche(HTRModel):
    # Needed for building bluche model
    def architecture(self, input_size=(1024, 128, 1), d_model=101):
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

        shape = cnn.shape
        blstm = Reshape((shape[1], shape[2] * shape[3]))(cnn)

        blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
        blstm = Dense(units=128, activation="tanh")(blstm)

        blstm = Bidirectional(LSTM(units=128, return_sequences=True))(blstm)
        output_data = Dense(units=d_model, activation="softmax")(blstm)

        return (input_data, output_data)

    def __init__(self, weigths_path="resources/bluche_bressay_weights.hdf5", charset_path="resources/bluche_bressay_charset.txt"):
        # Loads model's custom layers, optimizer and pretrained weigths
        (input_data, output_data) = self.architecture()
        self.model     = Model(input_data, output_data)
        self.optimizer = NormalizedOptimizer(
    tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.1)
)
        self.model.compile(optimizer=self.optimizer, loss=ctc_loss_lambda_func)
        self.model.load_weights(weigths_path)
        # Loads model charset for output decoding
        with open(charset_path,'r') as file:
            self.charset = file.read()

    def predict(self, image_list):
        # Get model output
        output = self.model.predict(image_list)

        # CTC decode
        #batch_size = int(np.ceil(len(output) / 1))
        input_length = len(max(output, key=len))

        predicts, probabilities = [], []

        #index = 0
        #until = batch_size
        #until = len(output)
        #x_test = np.asarray(output[index:until])
        x_test = np.asarray(output)
        x_test_len = np.asarray([input_length for _ in range(len(x_test))])
        decode, log = K.ctc_decode(x_test,x_test_len,greedy=False,beam_width=10,top_paths=1)
        decode = np.swapaxes(decode, 0, 1)
        predicts.extend([[[int(p) for p in x if p != -1] for x in y] for y in decode])
        probabilities.extend([np.exp(x) for x in log])
        predicts = [[self.decode(x) for x in y] for y in predicts]
        flat_predicts = [p[0] for p in predicts]
        return (flat_predicts, probabilities) # Retorna uma lista de strings preditas

    def evaluate(self, image_list, ground_truth_list):
        """
        Calcula CER e WER para uma lista de imagens e seus rótulos de verdade.
        """
        # 1. Obter as previsões
        predicted_texts, _ = self.predict(image_list)

        # 2. Aplicar pós-processamento, se houver (ex: removendo tokens especiais)
        # Note: A decodificação na sua função `predict` já remove "¶" e "¤".

        # 3. Calcular CER e WER (usando as funções definidas acima)
        # Certifique-se de que ground_truth_list e predicted_texts estejam limpos (sem pontuação excessiva para WER)
        cer, wer = calculate_cer_wer(ground_truth_list, predicted_texts)

        return cer, wer

# TODO: Testes de funcionalidade
"""if __name__=='__main__':
    # Build model
    model = Bluche()
    model.model.summary()
    # Build preprocessing
    import preprocess
    prep  = preprocess.Bluche()
    imgs  = prep.process([f'test/model/inputs/line{i+1}.png' for i in range(5)])
    predicts = model.predict(imgs)
    for i in range(5):
        print(f'LINHA {i+1}\n{predicts[0][i]}')

"""
if __name__=='__main__':

    # Exemplo (você precisará obter os ground_truth reais do seu dataset)
    import preprocess
    prep = preprocess.Bluche()
    model = Bluche()

    test_images = [f'test/model/inputs/line{i+1}.png' for i in range(5)]
    # Supondo que você tenha estas strings de ground truth:
    ground_truth_texts = [
        "reconhecimento igualitário da mulher somente ocorrerão, a longo prazo,",
        "a partir de quando o feminismo for pensado e discutido nas escolas e",
        "mídias, fazendo com que todos conscientizem-se sobre o assunto e assim,",
        "juntamente às opiniões pessoais, os valores nos quais a sociedade se pauta",
        "adequem-se aos conceitos mais humanitários."
    ]

    imgs = prep.process(test_images)

    # 1. Previsões originais (apenas para exibição)
    predicts, _ = model.predict(imgs)
    for i, pred in enumerate(predicts):
        print(f'LINHA {i+1} (GT: "{ground_truth_texts[i]}")\nPRED: {pred}\n')

    # 2. Avaliação usando a nova função
    cer, wer = model.evaluate(imgs, ground_truth_texts)

    print("\n--- MÉTICAS DE ERRO ---")
    print(f"Character Error Rate (CER): {cer:.2f}%")
    print(f"Word Error Rate (WER): {wer:.2f}%")
