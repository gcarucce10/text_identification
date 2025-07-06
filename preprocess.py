import os
from abc import ABC

import numpy as np
import cv2

# Resizes single-channel image to new specified dimensions (Scales by max factor
# with no distortion and fills remaining pixels with background uint8 color)
def resize_images(image_list, new_x, new_y):
    processed_images = []
    for image in image_list:
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        # Gets background info
        u, i = np.unique(np.array(image).flatten(), return_inverse=True)
        background = int(u[np.argmax(np.bincount(i))])
        # Finds max scale factor for no distortion
        h, w = np.asarray(image).shape
        wt, ht = new_x, new_y
        f = max((w / wt), (h / ht))
        new_size = (max(min(wt, int(w/f)), 1), max(min(ht, int(h/f)), 1))
        # Resizes with no distortion (completes one dimension with background color)
        image = cv2.resize(image, new_size)
        target = np.ones([ht, wt], dtype=np.uint8) * background
        target[0:new_size[1], 0:new_size[0]] = image
        #image = cv2.transpose(target)
        processed_images.append(target)
    return processed_images

# Normalizes single-channel image according to normal distribution (Z-Score)
# outputs list of np.float32 images
def normalize_images(image_list):
    processed_images = []
    for image in image_list:
        if isinstance(image, str):
            image = cv2.imread(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        mean, std  = np.mean(image), np.std(image)
        image = (image - mean) / std
        processed_images.append(image)
    return processed_images


# Abstract class for generic HTR preprocessing
class HTRPreprocessor(ABC):
    def __init__(self):
        pass

    # Recieves list of images as [np.array|path] and returns list of processed images as [np.array]
    # Standard Pipeline class assumes the preprocessing output and model input to have the SAME shape
    def process(self, image_list):
        pass


class Bluche(HTRPreprocessor):
    def __init__(self, input_size=(1024, 128, 1)):
        self.input_size=input_size
    
    def process(self, image_list):
        # Resizes to model input
        processed_images = resize_images(image_list, self.input_size[0], self.input_size[1])
        # Transposes
        processed_images = [cv2.transpose(image) for image in processed_images]
        # Normalizes
        processed_images = normalize_images(processed_images)
        # Reshape for CRNN input shape
        return np.expand_dims(np.asarray(processed_images).astype(np.float32), axis=-1)


# TODO: fix for windows file pathing
# # Functionality tests
if __name__=='__main__':
    original   = [cv2.imread(f'test/preprocess/inputs/original.png', cv2.IMREAD_GRAYSCALE), f'test/preprocess/inputs/original.png'] 
    resized    = resize_images(original, 1400, 100)
    #gray_scale      = to_gray_scale(resized)
    #rgb             = to_RGB(gray_scale)
    normalized = normalize_images(resized)
    #normalized_rgb  = normalize_image(rgb)
    # Prints (for value verification) and saves images in test output directory
    for case, image in zip(["resized", "normalized"], 
                           [resized, normalized]):
        print(f'{case}-img:  {image[0]}\n')
        print(f'{case}-path: {image[1]}\n')
        cv2.imwrite(f'test/preprocess/outputs/{case}-img.png',  image[0].astype(np.uint8))
        cv2.imwrite(f'test/preprocess/outputs/{case}-path.png', image[1].astype(np.uint8))
    # Tests Bluche preprocessor class
    processor = Bluche()
    results = processor.process(original)
    print(f'CLASS-img: \n{results[0]}')
    print(f'CLASS-path:\n{results[1]}')
    cv2.imwrite(f'test/preprocess/outputs/CLASS_OUTPUT-img.png',  results[0].astype(np.uint8))
    cv2.imwrite(f'test/preprocess/outputs/CLASS_OUTPUT-path.png', results[1].astype(np.uint8))