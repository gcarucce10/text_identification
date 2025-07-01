from PIL import Image
import numpy as np
from scipy.ndimage import zoom

def resize_image(input_image, new_x, new_y):
    """
    Redimensiona uma imagem para as dimensões especificadas.
    Parâmetros:
    - input_image: Caminho da imagem, objeto PIL.Image ou np.array
    - new_x: Nova largura desejada (em pixels)
    - new_y: Nova altura desejada (em pixels)
    
    Retorna:
    - np.array: Imagem redimensionada como array numpy
    """
    # Converter entrada para numpy array
    if isinstance(input_image, str):
        # Carregar imagem do arquivo
        imagem_pil = Image.open(input_image)
        image_array = np.array(imagem_pil)
    elif isinstance(input_image, Image.Image):
        # Converter PIL Image para numpy array
        image_array = np.array(input_image)
    elif isinstance(input_image, np.ndarray):
        # Já é um numpy array
        image_array = input_image
    else:
        raise ValueError("Tipo de entrada não suportado. Use string (caminho), PIL.Image ou np.array")
    
    # Obter dimensões atuais
    current_height, current_width = image_array.shape[:2]
    
    # Calcular fatores de escala
    scale_y = new_y / current_height
    scale_x = new_x / current_width
    
    # Aplicar redimensionamento
    if len(image_array.shape) == 3:  # Imagem colorida (altura, largura, canais)
        scale_factors = (scale_y, scale_x, 1)
    else:  # Imagem em escala de cinza (altura, largura)
        scale_factors = (scale_y, scale_x)
    
    # Usar zoom do scipy para redimensionar
    resized_image = zoom(image_array, scale_factors, order=3)  # order=3 para interpolação cúbica
    
    # Garantir que o tipo de dados seja preservado
    return resized_image.astype(image_array.dtype)


import numpy as np
from scipy.ndimage import affine_transform

def shear_image_centered(image_array, shear_x=0, shear_y=0):
  """
  Aplica um cisalhamento (shearing) a uma imagem representada por um np.array,
  ajustando as dimensões para conter toda a imagem transformada.

  Args:
    image_array (np.array): O array numpy representando a imagem.
    shear_x (float): O fator de cisalhamento na direção x.
    shear_y (float): O fator de cisalhamento na direção y.

  Returns:
    np.array: O array numpy da imagem cisalhada com dimensões ajustadas.
  """
  # Verificar se a imagem é colorida ou em escala de cinza
  is_color = len(image_array.shape) == 3
  
  if is_color:
    rows, cols, channels = image_array.shape

    # Determinante da matriz de cisalhamento
    det = 1 - shear_x * shear_y
    if abs(det) < 1e-10:
        raise ValueError("Shear values result in a singular matrix (cannot be inverted).")

    # Matriz de cisalhamento (forward transformation)
    shear_matrix = np.array([[1, shear_x],
                            [shear_y, 1]])

    # Calcular os cantos da imagem original
    corners = np.array([
        [0, 0],           # canto superior esquerdo
        [cols, 0],        # canto superior direito
        [0, rows],        # canto inferior esquerdo
        [cols, rows]      # canto inferior direito
    ])

    # Transformar os cantos para encontrar a bounding box
    transformed_corners = corners @ shear_matrix.T

    # Encontrar os limites da imagem transformada
    min_x = np.floor(np.min(transformed_corners[:, 0]))
    max_x = np.ceil(np.max(transformed_corners[:, 0]))
    min_y = np.floor(np.min(transformed_corners[:, 1]))
    max_y = np.ceil(np.max(transformed_corners[:, 1]))

    # Calcular novas dimensões
    new_cols = int(max_x - min_x)
    new_rows = int(max_y - min_y)

    # Matriz de cisalhamento inversa (para scipy)
    inv_shear_matrix = (1 / det) * np.array([[1, -shear_x],
                                            [-shear_y, 1]])

    # Calcular offset para posicionar a imagem na nova canvas
    if shear_x > 0:
        offset_x = -min_x
    else:
        offset_x = min_x
    if shear_y > 0:
        offset_y = -min_y
    else: 
        offset_y = min_y

    # Offset total para scipy (que usa transformação inversa)
    total_offset = np.array([offset_x, offset_y])

    # Criar array de resultado com as dimensões corretas
    result = np.zeros((new_rows, new_cols, channels), dtype=image_array.dtype)

    # Aplicar transformação em cada canal
    for channel in range(channels):
        result[:, :, channel] = affine_transform(
            image_array[:, :, channel],
            matrix=inv_shear_matrix,
            offset=total_offset,
            output_shape=(new_rows, new_cols),
            mode='constant',
            cval=0,
            order=3
        )

    return result.astype(image_array.dtype)


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
    imagem = shear_image_centered(imagem, shear_x, shear_y)
    # Aplica redimensionalizacao
    imagem = resize_image(imagem, new_size[0], new_size[1])
    # Aplica escala, se passado como parametro
    scalers_dict = {"standard": StandardScaler_image, "uniform": UniformScaler_image}
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
