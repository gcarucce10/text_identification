import numpy as np
import cv2
import math

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Multiply, Activation, Activation



#==============================================================================================================
# CUSTOM TOOLS/UTILITARIES for tensorflow models
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
#==============================================================================================================



#==============================================================================================================
# CUSTOM TOOLS/UTILITARIES for images
def createKernel(kernelSize, sigma, theta):
    # create anisotropic filter kernel according to given parameters
    assert kernelSize % 2       # must be odd size
    halfSize = kernelSize // 2  # get integer-valued resize of division

    kernel = np.zeros([kernelSize, kernelSize])  # kernel
    sigmaX = sigma          # scale factor for X dimension
    sigmaY = sigma * theta  # theta - multiplication factor = sigmaX/sigmaY

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x ** 2 / (2 * sigmaX) - y ** 2 / (2 * sigmaY))
            xTerm = (x ** 2 - sigmaX ** 2) / (2 * math.pi * sigmaX ** 5 * sigmaY)
            yTerm = (y ** 2 - sigmaY ** 2) / (2 * math.pi * sigmaY ** 5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel

def grayscale_image(image):
    if isinstance(image, str):
        image = cv2.imread(image, cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return target

# TODO: same as normalize_images in model.py
# Normalizes single-channel image according to normal distribution (Z-Score)
# outputs list of np.float32 images
def normalize_image(image):
    if isinstance(image, str):
        image = cv2.imread(image, cv2.COLOR_BGR2GRAY)
    target = image.astype(np.float32)
    mean, std  = np.mean(target), np.std(target)
    target = (target - mean) / std
    return target

# TODO: model.py usa coisa parecida (passar metodo pra la tbm)
def transpose_image(image):
    if isinstance(image, str):
        image = cv2.imread(image, cv2.COLOR_BGR2GRAY)
    target = np.transpose(image)
    return target

def filter_image(image, kernel, ddepth):
    if isinstance(image, str):
        image = cv2.imread(image, cv2.COLOR_BGR2GRAY)
    #TODO: Codigo original usa "borderType=cv2.BORDER_REPLICATE" (?)
    target = cv2.filter2D(src=image, kernel=kernel, ddepth=ddepth)
    return target

def smooth_image(image, window_len=11, window='hanning'):
    """ Image smoothing is achieved by convolving the image with a low-pass filter kernel.
    Such low pass filters as: ['flat', 'hanning', 'hamming', 'bartlett', 'blackman'] can be used
    It is useful for removing noise. It actually removes high frequency content
    (e.g: noise, edges) from the image resulting in edges being blurred when this is filter is applied."""
    if image.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return image
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[image[window_len-1:0:-1],image,image[-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y

def crop_lines(image, blanks):
    """ Splits the image with text into lines, according to the markup obtained from the created algorithm.
     Very first"""
    x1 = 0
    y = 0
    lines = []
    for i, blank in enumerate(blanks):
        x2 = blank
        line = image[:,  x1:x2]
        lines.append(line)
        x1 = blank
    #print("Lines found: {0}".format(len(lines)))
    return lines

# Resizes single-channel image to new specified dimensions (Scales by max factor
# with no distortion and fills remaining pixels with background uint8 color)
def resize_image(image, new_x, new_y):
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # Gets background info
    u, i = np.unique(np.array(image).flatten(), return_inverse=True)
    background = int(u[np.argmax(np.bincount(i))])
    # Finds max scale factor for no distortion
    #print(np.asarray(image).shape)
    h, w = np.asarray(image).shape
    wt, ht = new_x, new_y
    f = max((w / wt), (h / ht))
    new_size = (max(min(wt, int(w/f)), 1), max(min(ht, int(h/f)), 1))
    # Resizes with no distortion (completes one dimension with background color)
    target = np.ones([ht, wt], dtype=np.uint8) * background
    target[0:new_size[1], 0:new_size[0]] = cv2.resize(image, new_size)
    return target
#==============================================================================================================



#==============================================================================================================
# CUSTOM TOOLS/UTILITARIES for strings
def join_lines(line_list):
    return "\n".join([line for line in line_list])

def levenshtein_distance(s1: list, s2: list) -> int:
    """Calcula a distância de Levenshtein (edição) entre duas listas de tokens."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, token1 in enumerate(s1):
        current_row = [i + 1]
        for j, token2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (token1 != token2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_cer_wer(ground_truth_list: list, predicted_list: list):
    """
    Calcula o Character Error Rate (CER) e Word Error Rate (WER).
    
    Args:
        ground_truth_list: Lista de strings de referência.
        predicted_list: Lista de strings preditas.
        
    Returns:
        Um tuple (CER, WER).
    """
    total_cer_dist = 0
    total_wer_dist = 0
    total_chars = 0
    total_words = 0

    for gt_text, pred_text in zip(ground_truth_list, predicted_list):
        # 1. Cálculo do CER (comparação caractere por caractere)
        gt_chars = list(gt_text)
        pred_chars = list(pred_text)
        total_cer_dist += levenshtein_distance(gt_chars, pred_chars)
        total_chars += len(gt_chars)

        # 2. Cálculo do WER (comparação palavra por palavra)
        gt_words = gt_text.split()
        pred_words = pred_text.split()
        total_wer_dist += levenshtein_distance(gt_words, pred_words)
        total_words += len(gt_words)
        
    # Evita divisão por zero
    cer = (total_cer_dist / total_chars) * 100 if total_chars > 0 else 0
    wer = (total_wer_dist / total_words) * 100 if total_words > 0 else 0

    return cer, wer
#==============================================================================================================