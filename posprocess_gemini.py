import os
import re
import requests
import language_tool_python

# Inicializa LanguageTool local para pt-BR
tool = language_tool_python.LanguageTool('pt-BR')

# Carrega dicionário simples (palavras PT‑BR)
with open('palavras_ptbr.txt', encoding='utf-8') as f:
    valid_words = set(line.strip().lower() for line in f if line.strip())

# 1. Função para corrigir com LanguageTool
def corrigir_lt(texto: str) -> str:
    matches = tool.check(texto)
    return language_tool_python.utils.correct(texto, matches)

# 2. Detecta se ainda há erros significativos
def precisa_gemini(texto: str) -> bool:
    # 1. Se LT ainda detectou erros residuais
    if tool.check(texto):
        return True

    # 2. Se tem palavra desconhecida
    for w in texto.split():
        w_clean = w.lower().strip('.,;:!?')
        if w_clean.isalpha() and w_clean not in valid_words:
            return True

    # 3. Padrões OCR suspeitos (letras duplicadas)
    if re.search(r'\b\w*(\w)\1{2,}\w*\b', texto):
        return True

    return False


# 3. Chama Gemini via REST (sem SDK)
def corrige_com_gemini(texto: str) -> str:
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

# 4. Função híbrida de correção
def corrigir_hibrido(texto: str) -> str:
    lt = corrigir_lt(texto)
    print("Texto corrigido pelo LanguageTool:", lt)
    if precisa_gemini(lt):
        print("Precisou de correção adicional com Gemini.")
        return corrige_com_gemini(lt)
    return lt

# 5. Processa lista de frases
def processar_lista(lista):
    return [corrigir_hibrido(frase) for frase in lista]

def carregar_frases(caminho_arquivo):
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        # Lê todas as linhas, remove espaços em excesso e aspas
        return [linha.strip().strip('"') for linha in f if linha.strip()]

if __name__ == "__main__":
    
    teste = "do pópio moda. É evidento que embora a situaçãeo esteja melhor em"
    print("Texto original:", teste)
    print("Texto corrigido:", corrigir_hibrido(teste))
    
    # testes = carregar_frases("teste.txt")
    
    # for t in testes:
    #     print("Antes:", t)
    #     print("Depois:", corrigir_hibrido(t), "\n")
