import numpy as np
import cv2

# Transforms image into single-channel grayscale image 
def to_gray_scale(input_image):
    return cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Transforms image into 3-channel image
def to_RGB(input_image):
    return np.stack((input_image,)*3, axis=-1)

# Resizes image to new specified x and y dimensions
def resize_image(input_image, new_x, new_y):
    return cv2.resize(input_image, dsize=(new_x, new_y))

# Normalizes image according to normal distribution
def normalize_image(input_image):
    image = input_image.astype(np.float32)
    mean  = np.mean(image)
    std   = np.std(image)
    return (image - mean) / std


#TODO: different generalization?
# steps_dict = dict("key": [func1, func2...])
# param_dict = dict("key": [{params1}, {params2}])
# for ...
#   img = steps_dict["key"][i](img, **param_dict["key"][i])
# Main class for Handwritten Text Recognition Preprocessing
class HTRPreprocessor:
    # Dictionary of process() params for easy selecting
    params_dict = {
        "standard": dict(new_size=(1024,128,1), normalize = True, paragraph = False, astype=np.float32)
    }

    def __init__(self, params="standard"):
        if isinstance(params, str):
            self.params = self.params_dict[params]
        elif isinstance(params, dict):
            self.params = params

    # TODO: Handle paragraph case with line segmentation
    # Recieves list of images (as [np.array]) and return list of processed images, (as [np.array.astype(self.astype)])
    # Preprocesses according to params specified in builder (self.params_dict or custom dict as input)
    def process(self, image_list):
        processed_images = []
        for image in image_list:
            if isinstance(image, str):
                image = cv2.imread(image)
            # Handles change in channels
            if self.params["new_size"][2] == 1:
                image = to_gray_scale(image)
            elif self.params["new_size"][2] == 3:
                image = to_RGB(image)
            # Standard preprocessing
            image = resize_image(image, self.params["new_size"][0], self.params["new_size"][1])
            if self.params["normalize"]:
                image = normalize_image(image)
            processed_images.append(image.astype(self.params["astype"]))
        return processed_images


# TODO: fix for windows file pathing
# # Functionality tests
if __name__=='__main__':
    # Loads test image and makes non-destructive transforms
    original        = cv2.imread(f'test/preprocess/inputs/original.png')
    resized         = resize_image(original, 750, 50)
    gray_scale      = to_gray_scale(resized)
    rgb             = to_RGB(gray_scale)
    normalized_gray = normalize_image(gray_scale)
    normalized_rgb  = normalize_image(rgb)
    # Prints (for value verification) and saves images in test output directory
    for case, image in zip(["resized", "gray", "rgb", "normalized_gray", "normalized_rgb"], 
                           [resized, gray_scale, rgb, normalized_gray, normalized_rgb]):
        print(f'{case}: {image}\n')
        image = image.astype(np.uint8)
        cv2.imwrite(f'test/preprocess/outputs/{case}.png', image)
    # Tests HTRPreprocessor class
    processor = HTRPreprocessor("standard")
    results = processor.process([original])
    print(f'CLASS\n{results[0]}')
    results[0] = results[0].astype(np.uint8)
    cv2.imwrite(f'test/preprocess/outputs/CLASS_OUTPUT.png', results[0])