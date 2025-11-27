import os
import re
import requests
from autocorrect import Speller

# Disables tensorflow usual debbug messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



#==============================================================================================================
# CUSTOM CLASSES FOR PERFORMING POSTPROCESSING
# Class for autocorrect library
class Autocorrect():
    def __init__(self, language='pt'):
        self.spell = Speller(language, only_replacements=True)
    
    def preprocess(self, text):
        return text
    
    def correct(self, text):
        return self.spell(text)

# Class for gemini API
class GEMINI_API():
    def __init__(self, API_key = None, prompt = None, URL = None, headers = None, params = None):
        if API_key == None:
            API_key = os.getenv("GEMINI_API_KEY")  # Google comments this
        self.API_key = API_key

        if prompt == None:
            #prompt = (
            #    "Você é um corretor profissional de português brasileiro (pt-BR).\n"
            #    "Corrija apenas ortografia, acentuação e gramática da frase abaixo, "
            #    "respondendo somente com a frase corrigida, sem explicações\n\n"
            #)
            prompt = (
                "Performe OCR (optimal character recognition) na frase abaixo, em português"
                "pt-BR, respondendo somente com a frase corrigida, sem explicações\n\n"
            )
        self.prompt = prompt

        if URL == None:
            URL = (
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-2.5-flash:generateContent"
            )
        self.URL = URL

        if headers == None:
            headers = {"Content-Type": "application/json"}
        self.headers = headers

        if params == None:
            params = {"key": self.API_key}
        self.params = params

    def preprocess(self, text):
        return text

    def correct(self, text):
        body = {
            "contents": [
                { "parts": [ { "text": self.prompt + text } ] }
            ]
        }
        resp = requests.post(self.URL, headers=self.headers, params=self.params, json=body)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
#==============================================================================================================



#==============================================================================================================
# POSTPROCESS CLASSES DICTIONARY FOR EASY REFERENCING
postprocess_dict = {
    'auto': Autocorrect,
    'gemini': GEMINI_API
}
#==============================================================================================================



# Unit tests for postprocess.py
if __name__ == "__main__":
    for key, value in postprocess_dict.items():
        model = value()
        for i in range(11):        
            with open(f'tests/postprocess/inputs/linha{i}.txt','r') as f:
                text = f.read()
            with open(f'tests/postprocess/outputs/{key}-linha{i}.txt','w') as f:
                f.write(model.correct(model.preprocess(text)))