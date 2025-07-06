import os
from abc import ABC
import re
import requests
import language_tool_python

# Initializes local LanguageTool for pt-BR 
tool = language_tool_python.LanguageTool('pt-BR')

# Loads simple word dictionary (pt‑BR words)
with open('resources/palavras_ptbr.txt', encoding='utf-8') as f:
    valid_words = set(line.strip().lower() for line in f if line.strip())

# 1. Function for correcting with LanguageTool
def correct_lt(text: str) -> str:
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)

# 2. Detects existence of meaningful errors
def needs_gemini(text: str) -> bool:
    # 1. If LT didn't detect residual errors
    if tool.check(text):
        return True

    # 2. If unknown words exist
    for w in text.split():
        w_clean = w.lower().strip('.,;:!?')
        if w_clean.isalpha() and w_clean not in valid_words:
            return True

    # 3. Suspect OCR patterns (dupped letter)
    if re.search(r'\b\w*(\w)\1{2,}\w*\b', text):
        return True

    return False

# 3. Calls Gemini via REST (no SDK)
def correct_with_gemini(texto: str) -> str:
    prompt = (
        "Você é um corretor profissional de português brasileiro (pt-BR).\n"
        "Corrija apenas ortografia, acentuação e gramática da frase abaixo.\n"
        "Responda somente com a frase corrigida, sem explicações.\n\n"
        f'Frase: "{texto}"'
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
    )
    headers = {"Content-Type": "application/json"}
    params = {"key": os.getenv("GEMINI_API_KEY")}
    body = {
        "contents": [
            { "parts": [ { "text": prompt } ] }
        ]
    }

    resp = requests.post(url, headers=headers, params=params, json=body)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

# 4. Hybrid correcting function (LanguageTool and, if needed, Gemini)
def correct_hybrid(texto: str) -> str:
    lt = correct_lt(texto)
    #print("Texto corrigido pelo LanguageTool:", lt)
    if needs_gemini(lt):
        #print("Precisou de correção adicional com Gemini.")
        return correct_with_gemini(lt)
    return lt


class HTRPostprocessing:
    # Dictionary of process() method for easy selecting
    method_dict = {
        "dict":   correct_lt,
        "llm":    correct_with_gemini,
        "hybrid": correct_hybrid
    }
    
    def __init__(self, method="hybrid"):
        self.postprocessor = self.method_dict[method]

    # 
    def process(self, text_list):
        return [self.postprocessor(text) for text in text_list]


# Functionality tests
if __name__ == "__main__":
    def carregar_frases(caminho_arquivo):
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            # Lê todas as linhas, remove espaços em excesso e aspas
            return [linha.strip().strip('"') for linha in f if linha.strip()]

    teste = "do pópio moda. É evidento que embora a situaçãeo esteja melhor em"
    print("Texto original:", teste)
    print("Texto corrigido:", correct_hybrid(teste))
    
    # testes = carregar_frases("teste.txt")
    
    # for t in testes:
    #     print("Antes:", t)
    #     print("Depois:", corrigir_hibrido(t), "\n")