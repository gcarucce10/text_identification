from PIL import Image

def resize_image(imagem_entrada, new_x, new_y):
    """
    Redimensiona uma imagem para as dimensões especificadas.
    Parâmetros:
    - imagem_entrada: Caminho da imagem ou objeto PIL.Image
    - new_x: Nova largura desejada (em pixels)
    - new_y: Nova altura desejada (em pixels)
    """
    # Abrir a imagem se for uma string (caminho)
    if isinstance(imagem_entrada, str):
        imagem = Image.open(imagem_entrada)
    else:
        imagem = imagem_entrada

    imagem_redimensionada = imagem.resize((new_x, new_y), Image.ANTIALIAS)
    return imagem_redimensionada


from PIL import Image

def shear_imagem(imagem_entrada, shear_x=0.0, shear_y=0.0):
    """
    Aplica uma transformação de cisalhamento (shear) à imagem.
    Parâmetros:
    - imagem_entrada: Caminho da imagem ou objeto PIL.Image
    - shear_x: Fator de cisalhamento horizontal (em radianos ou tangente do ângulo desejado)
    - shear_y: Fator de cisalhamento vertical (idem)
    """
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