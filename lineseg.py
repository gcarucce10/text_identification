import numpy as np
import scipy
import cv2
# Importing image manipulation utils
from utils import createKernel, grayscale_image, normalize_image
from utils import transpose_image, filter_image, smooth_image, crop_lines



#==============================================================================================================
# CUSTOM CLASSES FOR PERFORMING LINE SEGMENTATION
# Based on Irina's implementation of "Scale space technique for lines segmentation" proposed by R. Manmatha
class Irina():
    def __init__(self, kernelSize=25, sigma=11, theta=7, ddepth=-1):
        self.kernelSize = kernelSize
        self.sigma  = sigma
        self.theta  = theta
        self.ddepth = ddepth
        self.kernel = createKernel(self.kernelSize, self.sigma, self.theta)

    def preprocess(self, image):
        return image

    def segment(self, image):
        # Gets grayscale transposed image
        transposed_img  = grayscale_image(image)
        transposed_img  = transpose_image(transposed_img)
        # Gets normalized filtered image
        filtered_img = normalize_image(transposed_img)
        filtered_img = filter_image(filtered_img, self.kernel, self.ddepth)
        filtered_img = normalize_image(filtered_img)
        # Make summ elements in columns to get function of pixels value for each column
        summ_pix = np.sum(filtered_img, axis = 0)
        smoothed = smooth_image(summ_pix, 35)
        mins     = np.array(scipy.signal.argrelmin(smoothed, order=2))
        # Returns transposed (original orientation) lines
        found_lines = [transpose_image(_) for _ in crop_lines(transposed_img, mins[0])]
        return found_lines
#==============================================================================================================



#==============================================================================================================
# LINE SEGMENTATION CLASSES DICTIONARY FOR EASY REFERENCING
lineseg_dict = {
    'irina': Irina
}
#==============================================================================================================



# Unit tests for lineseg.py
if __name__=='__main__':
    img = 'tests/lineseg/inputs/carucce.jpeg'
    lineseg = Irina()
    for i, line in enumerate(lineseg.segment(lineseg.preprocess(img))):
        cv2.imwrite(f'tests/lineseg/outputs/line{i}.png',line)