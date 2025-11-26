# Disables tensorflow usual debbug messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from lineseg     import lineseg_dict
from HTR         import HTR_dict
from postprocess import postprocess_dict

from utils import join_lines

import cv2



#==============================================================================================================
# CUSTOM CLASSES FOR RUNNING FULL PIPELINES
# General class for line-level text recognition
class line_level_pipeline():
    def __init__(self, lineseg, HTR, postprocess=None, lineseg_params={}, HTR_params={}, postprocess_params={}):
        # Initializes class instances if using the dictionaries
        if isinstance(lineseg,str):
            lineseg     = lineseg_dict[lineseg](**lineseg_params)
        if isinstance(HTR,str):
            HTR         = HTR_dict[HTR](**HTR_params)
        if isinstance(postprocess,str):
            postprocess = postprocess_dict[postprocess](**postprocess_params)
        self.lineseg, self.HTR, self.postprocess = lineseg, HTR, postprocess

    def run(self, image_list):
        text_list = []
        for image in image_list:
            # Gets segmented lines for the image of a single text
            segmented_lines = self.lineseg.segment(self.lineseg.preprocess(image))
            # For each segmented line, transcribes using HTR model and postprocess (if defined)
            transcribed_lines = []
            for line in segmented_lines:
                transcript = self.HTR.predict(self.HTR.preprocess(line))
                if self.postprocess is not None:
                    transcript = self.postprocess.correct((self.postprocess.preprocess(transcript)))
                transcribed_lines.append(transcript)
            # Joins lines into a single text and appends to text_list
            text_list.append(join_lines(transcribed_lines))
        return text_list
#==============================================================================================================



#==============================================================================================================
# PIPELINE CLASSES DICTIONARY FOR EASY REFERENCING
pipeline_dict = {
    'line': line_level_pipeline
}
#==============================================================================================================



# Unit tests for HTR.py
if __name__=="__main__":
    pipeline = line_level_pipeline('irina', 'bluche', 'gemini')
    text = pipeline.run(['tests/pipeline/inputs/paragraph.png'])
    with open(f'tests/pipeline/outputs/paragraph.txt', 'w') as f:
        f.write(text[0])