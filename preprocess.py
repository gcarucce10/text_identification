from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def resize_image(imagem_entrada, new_x, new_y):
    '''
    Redimensiona uma imagem para as dimensões especificadas.
    Parâmetros:
    - imagem_entrada: Caminho da imagem ou objeto PIL.Image
    - new_x:          Nova largura desejada (em pixels)
    - new_y:          Nova altura desejada (em pixels)
    '''
    # Abrir a imagem se for uma string (caminho)
    if isinstance(imagem_entrada, str):
        imagem = Image.open(imagem_entrada)
    else:
        imagem = imagem_entrada

    imagem_redimensionada = imagem.resize((new_x, new_y), Image.ANTIALIAS)
    return imagem_redimensionada

def shear_image(imagem_entrada, shear_x=0.0, shear_y=0.0):
    '''
    Aplica uma transformação de cisalhamento (shear) à imagem.
    Parâmetros:
    - imagem_entrada: Caminho da imagem ou objeto PIL.Image
    - shear_x:        Fator de cisalhamento horizontal (em radianos ou tangente do ângulo desejado)
    - shear_y:        Fator de cisalhamento vertical (idem)
    '''
    # Abrir a imagem se for um caminho
    if isinstance(imagem_entrada, str):
        imagem = Image.open(imagem_entrada)
    else:
        imagem = imagem_entrada

    largura, altura = imagem.size

    # Matriz de transformação afim (6 valores):
    # (a, b, c, d, e, f) → aplica a transformação:
    # x' = a*x + b*y + c
    # y' = d*x + e*y + f
    matriz_afim = (
        1, shear_x, 0,  # primeira linha: afeta o eixo X
        shear_y, 1, 0   # segunda linha: afeta o eixo Y
    )

    # Calcula novo tamanho para evitar corte da imagem
    nova_largura = int(largura + abs(shear_x) * altura)
    nova_altura = int(altura + abs(shear_y) * largura)

    imagem_cisalhada = imagem.transform(
        (nova_largura, nova_altura),
        Image.AFFINE,
        matriz_afim,
        resample=Image.BICUBIC,
        fillcolor=(255, 255, 255)  # Fundo branco
    )

    return imagem_cisalhada

def StandardScaler_image(imagem_entrada):
    '''
    Escala os valores dos pixels da imagem de acordo com a distribuicao normal.
    Parâmetros:
    - imagem_entrada: Caminho da imagem ou objeto PIL.Image
    '''
    # Abrir a imagem se for um caminho
    if isinstance(imagem_entrada, str):
        imagem = Image.open(imagem_entrada)
    else:
        imagem = imagem_entrada
    
    imagem = np.asarray(imagem)
    scaler = StandardScaler().fit(imagem)
    imagem_escalada = Image.fromarray(scaler.transform(imagem))
    
    return imagem_escalada

def UniformScaler_image(imagem_entrada):
    '''
    Escala os valores dos pixels da imagem de acordo com a distribuicao uniforme ([0,255] -> [0,1]).
    Parâmetros:
    - imagem_entrada: Caminho da imagem ou objeto PIL.Image
    '''
    # Abrir a imagem se for um caminho
    if isinstance(imagem_entrada, str):
        imagem = Image.open(imagem_entrada)
    else:
        imagem = imagem_entrada
    
    imagem = np.asarray(imagem)
    scaler = MinMaxScaler().fit(imagem)
    imagem_escalada = Image.fromarray(scaler.transform(imagem))
    
    return imagem_escalada
    
def StandardPreprocess_image(imagem_entrada, new_size=None, shear_x=0.0, shear_y=0.0, scaler=None):
    '''
    Preprocessamento padrao para uma unica imagem
    Parâmetros:
    - imagem_entrada: Caminho da imagem ou objeto PIL.Image
    - new_size:       Nova dimensao desejada em pixels (tuple(new_x, new_y))
    - shear_x:        Fator de cisalhamento horizontal (em radianos ou tangente do ângulo desejado)
    - shear_y:        Fator de cisalhamento vertical (idem)
    - scaler:         str com o metodo de escala desejado ("uniform"; "standard")
    '''
    # Abrir a imagem se for um caminho
    if isinstance(imagem_entrada, str):
        imagem = Image.open(imagem_entrada)
    else:
        imagem = imagem_entrada
    # Se dimensao for preservada, salva as dimensoes originais (cisalhamento pode alterar)
    if new_size == None:
        new_size = imagem.size
    # Aplica cisalhamento
    imagem = shear_image(imagem, shear_x, shear_y)
    # Aplica redimensionalizacao
    imagem = resize_image(imagem, new_size[0], new_size[1])
    # Aplica escala, se passado como parametro
    scalers_dict = {"standard"=StandardScaler_image, "uniform"=UniformScaler_image}
    if scaler in [key for key in scalers_dict.keys()]:
        imagem = scalers_dict[scaler](imagem)
    
    return imagem

def StandardPreprocess_imagelist(vetor_entrada, new_size=None, shear_x=0.0, shear_y=0.0, scaler=None):
    '''
    Preprocessamento padrao para uma lista de imagens
    Parâmetros:
    - imagem_entrada: Caminho da imagem ou objeto PIL.Image
    - new_size:       Nova dimensao desejada em pixels (tuple(new_x, new_y))
    - shear_x:        Fator de cisalhamento horizontal (em radianos ou tangente do ângulo desejado)
    - shear_y:        Fator de cisalhamento vertical (idem)
    - scaler:         str com o metodo de escala desejado ("uniform", "standard")
    '''

    vetor = []
    for imagem in vetor_entrada:
        vetor.append(StandardPreprocess_image(imagem, new_size, shear_x, shear_y, scaler))
    return vetor

# TODO: os retornos do preprocessamento devem ser PIL.Image, np.array ou tensor?
# TODO: Testes de funcionalidade
if __name__=='__main__':
    pass
