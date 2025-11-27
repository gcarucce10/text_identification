from xml.parsers.expat import model # remove after creating tests.py

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, Dense
from tensorflow.keras.layers import Input, MaxPooling2D, Reshape

# Importing tensorflow model utils
from utils import GatedConv2D, NormalizedOptimizer, ctc_loss_lambda_func
# Importing image manipulation utils
from utils import resize_image, normalize_image, transpose_image, grayscale_image



#==============================================================================================================
# CUSTOM CLASSES FOR PERFORMING HTR
# Class for bluche model trained on bressay
class Bluche_BRESSAY():
    # Needed for building bluche model
    def architecture(self):
        """
        Gated Convolucional Recurrent Neural Network by Bluche et al.

        Reference:
            Bluche, T., Messina, R.:
            Gated convolutional recurrent neural networks for multilingual handwriting recognition.
            In: Document Analysis and Recognition (ICDAR), 2017
            14th IAPR International Conference on, vol. 1, pp. 646–651, 2017.
            URL: https://ieeexplore.ieee.org/document/8270042
        """
        input_size, d_model = self.input_size, self.d_model

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

        return (input_data, output_data)

    def __init__(self, weigths_path="resources/bluche_bressay_weights.hdf5",
                 charset_path="resources/bluche_bressay_charset.txt",
                 input_size=(1024, 128, 1), d_model=101):
        # Loads model's custom layers, optimizer and pretrained weigths
        self.input_size, self.d_model = input_size, d_model
        (input_data, output_data) = self.architecture()
        self.model     = Model(input_data, output_data)
        self.optimizer = NormalizedOptimizer(tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.1))
        self.model.compile(optimizer=self.optimizer, loss=ctc_loss_lambda_func)
        self.model.load_weights(weigths_path)
        # Loads model charset for output decoding
        with open(charset_path,'r') as file:
            self.charset = file.read()

    def preprocess(self, image):
        # Resizes image to model input
        processed_image = resize_image(image, self.input_size[0], self.input_size[1])
        # Transposes image
        processed_image = transpose_image(processed_image)
        # Normalizes image
        processed_image = normalize_image(processed_image)
        # Reshape for CRNN input shape
        return np.expand_dims(np.asarray([processed_image]).astype(np.float32), axis=-1)

    def predict(self, image):
        # Get model output
        output = self.model.predict(image,verbose = 0)

        # CTC decode
        input_length = len(max(output, key=len))

        predicts, probabilities = [], []

        x_test = np.asarray(output)
        x_test_len = np.asarray([input_length for _ in range(len(x_test))])
        decode, log = K.ctc_decode(x_test,x_test_len,greedy=False,beam_width=10,top_paths=1)
        decode = np.swapaxes(decode, 0, 1)
        predicts.extend([[[int(p) for p in x if p != -1] for x in y] for y in decode])
        probabilities.extend([np.exp(x) for x in log])
        predicts = [[self.decode(x) for x in y] for y in predicts]
        flat_predicts = [p[0] for p in predicts]
        # TODO: return somente a mais provavel
        return flat_predicts[0] # Retorna uma lista de strings preditas

    def decode(self, text, pad="¶", unk="¤"):
        decoded = "".join([self.charset[int(x)] for x in text if x > -1])
        decoded = decoded.replace(pad, "").replace(unk, "")
        return decoded

    #TODO: tirar o evaluate daqui
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
#==============================================================================================================



#==============================================================================================================
# HTR CLASSES DICTIONARY FOR EASY REFERENCING
HTR_dict = {
    'bluche': Bluche_BRESSAY
}
#==============================================================================================================



# Unit tests for HTR.py
if __name__=='__main__':
    for key, value in HTR_dict.items():
        model = value()
        for i in range(13):  
            with open(f'tests/HTR/outputs/{key}-linha{i}.txt', 'w') as f:
                f.write(model.predict(model.preprocess(f'tests/HTR/inputs/linha{i}.png')))