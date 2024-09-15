import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from skimage.morphology import skeletonize

DEBUG_MODE = True
SMALL_COMPONENT_MIN_AREA = 80

def plot_images(images_titles):
    num_images = len(images_titles)
    fig, axs = plt.subplots(1, num_images, figsize=(10, 10))

    fig.patch.set_facecolor('#323232')  # Cor de fundo da figura (cinza escuro)

    if num_images == 1:
        axs = [axs]  # para garantir que axs seja uma lista mesmo com uma única imagem

    for i, (title, img) in enumerate(images_titles.items()):
        # Verifica se a imagem é RGB ou escala de cinza
        if len(img.shape) == 2:  # Verifica se é uma imagem em tons de cinza
            axs[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            # Converte para RGB se for uma imagem BGR (carregada pelo OpenCV)
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            axs[i].imshow(img_rgb)

        axs[i].set_title(title)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

def invert_channel(channel):
    return abs(255 - channel)

def binarize_image(image):
    # Converter para HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # Inverter o canal S para melhorar a binarização
    inverted_s = invert_channel(hsv[:, :, 1])

    if (DEBUG_MODE):
        # Mostrar os canais H, S e V
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('#323232')  # Cor de fundo da figura (cinza escuro)
        titles = ['H', 'S', 'V']
        for i in range(3):
            axs[i].imshow(hsv[:, :, i], cmap='gray', vmin=0, vmax=255)
            axs[i].set_title(titles[i])
        plt.show()

    # Binarizar o canal H utlizando OTSU
    _, binarized_H = cv.threshold(hsv[:, :, 0], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Binarizar o canal S utilizando threshold
    threshold = 210
    _, binarized_S = cv.threshold(inverted_s, threshold, 255, cv.THRESH_BINARY)

    # Retornar o and entre os dois resultados
    return cv.bitwise_and(binarized_H, binarized_S)

def convert_to_gray(img):
    # Função para converter a imagem RGB para escala de cinza usando a fórmula dada
    gray = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray[i, j] = 0.229 * img[i, j, 2] + 0.587 * img[i, j, 1] + 0.114 * img[i, j, 0]
    return gray

# Extrai apenas os elementos conectados com a borda
def connect_to_border(image):
    # Fazer dilatação na imagem
    kernel = np.ones((2, 2), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)

    # Criar uma máscara com zeros
    mask = np.zeros_like(image, dtype=np.uint8)

    # Dimensões da imagem
    rows, cols = image.shape

    # Criar uma fila para o BFS
    queue = deque()

    # Adicionar todos os pixels das bordas à fila
    for i in range(rows):
        if image[i, 0] == 255:
            queue.append((i, 0))
            mask[i, 0] = 255
        if image[i, cols - 1] == 255:
            queue.append((i, cols - 1))
            mask[i, cols - 1] = 255

    for j in range(cols):
        if image[0, j] == 255:
            queue.append((0, j))
            mask[0, j] = 255
        if image[rows - 1, j] == 255:
            queue.append((rows - 1, j))
            mask[rows - 1, j] = 255

    # Direções para explorar os vizinhos (8 direções)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # BFS para preencher a máscara
    while queue:
        x, y = queue.popleft()

        # Verificar todos os 8 vizinhos
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Verifica se o vizinho está dentro dos limites da imagem
            if 0 <= nx < rows and 0 <= ny < cols:
                # Se o vizinho for branco na imagem e ainda não estiver marcado na máscara
                if image[nx, ny] == 255 and mask[nx, ny] == 0:
                    queue.append((nx, ny))  # Adicionar o vizinho à fila
                    mask[nx, ny] = 255      # Marcar o vizinho na máscara como conectado
    return mask

# Remove ruídos (elementos conexos muito pequenos)
def remove_small_components(image, min_area):
    # Detectar componentes conectados
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(image, connectivity=8)

    # Crie uma máscara para os componentes maiores que min_area
    cleaned_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(1, num_labels):  # Ignorar o fundo (label 0)
        area = stats[i, cv.CC_STAT_AREA]
        if area >= min_area:
            cleaned_image[labels == i] = 255

    return cleaned_image

def close_small_holes(image):
    image_size = np.sqrt(image.shape[0] * image.shape[1])
    min_area = image_size // 2     #min_area = 170

    # Encontra os componentes conectados na imagem binária
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(~image, connectivity=8)  # Invertendo a imagem (buracos são componentes conectados)
    
    # Cria uma máscara para os buracos pequenos a serem fechados
    small_holes_mask = np.zeros_like(image, dtype=np.uint8)
    
    # Loop para cada componente conectado (buraco)
    for i in range(1, num_labels):  # Começa em 1 para pular o background (0)
        area = stats[i, cv.CC_STAT_AREA]
        # Se a área do componente conectado for menor que min_area, fechamos o buraco
        if area < min_area:
            # Adiciona o componente à máscara dos buracos pequenos
            small_holes_mask[labels == i] = 255
    
    # Adiciona a máscara de buracos pequenos à imagem original para fechar os buracos
    closed_image = cv.bitwise_or(image, small_holes_mask)
    
    return closed_image

def morphological_gradient(image):
    kernel = np.ones((2, 2), np.uint8)
    gradient = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)
    return gradient

# Função para calcular a razão área/perímetro
def calculate_ratio(contour):
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return np.sqrt(area) / perimeter

#Funcao para detectar rodovias pelas formas geometricas
def detect_road(closed_holes_image):
    # Aplicar um fechamento para unir regiões próximas
    kernel = np.ones((5, 5), np.uint8)
    closing_nossa = cv.morphologyEx(closed_holes_image, cv.MORPH_CLOSE, kernel)

    # Encontrar os contornos das regiões brancas
    contours_nossa, _ = cv.findContours(closing_nossa, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Filtrar os contornos que podem representar rodovias
    min_ratio = 0.1  # Ajuste esse valor conforme necessário

    road_contours = [cnt for cnt in contours_nossa if calculate_ratio(cnt) < min_ratio]

    #print (f'Número de rodovias detectadas: {len(road_contours)}')

    # Criar uma imagem de saída
    output = np.zeros_like(closed_holes_image)
    cv.drawContours(output, road_contours, -1, 255, thickness=cv.FILLED)

    output = cv.subtract(closed_holes_image, output)
    output2 = cv.subtract(closed_holes_image, output)

    return output2

def road_contour_line(original_image, roads):
    gradient = morphological_gradient(roads)
    gray_image = convert_to_gray(original_image)
    contour_line_image = cv.bitwise_or(gray_image, gradient)
    contour_line_image = cv.cvtColor(contour_line_image, cv.COLOR_GRAY2BGR)
    return gradient, contour_line_image

def skeletonize_image(original_image, roads):
    # Normalizar a imagem para 0 e 1 (necessário para a função skeletonize)
    roads = roads // 255
    skeleton = skeletonize(roads)
    # Converter de volta para 0-255 para exibição com o OpenCV/Matplotlib
    skeleton = (skeleton * 255).astype(np.uint8)
    # Botar o esqueleto por cima da imagem original
    gray_image = convert_to_gray(original_image)
    skeleton_image = cv.bitwise_or(gray_image, skeleton)
    skeleton_image = cv.cvtColor(skeleton_image, cv.COLOR_GRAY2BGR)
    return skeleton, skeleton_image

def main():
    img = cv.imread('../Imagens/imagem-teste2.png', cv.IMREAD_COLOR)
    # Binarizar a imagem
    thresholded_image = binarize_image(img) 

    # Remover pequenos componentes conexos (ajustar constante `min_area` conforme necessário)
    cleaned_image = remove_small_components(thresholded_image, SMALL_COMPONENT_MIN_AREA)

    # Pegar apenas elementos conexos a borda
    road_with_noise = connect_to_border(cleaned_image)

    # Dilatar a imagem para preencher buracos
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv.dilate(road_with_noise, kernel, iterations=1)

    # Fechar buracos pequenos
    closed_holes_image = close_small_holes(dilated_image)

    # Detectar rodovias pelas formas geométricas
    roads = detect_road(closed_holes_image)

    # Gerando imagem com as bordas
    gradient, contour_line_image = road_contour_line(img, roads)
    
    # Gerando imagem esqueletizada
    skeleton, skeleton_image = skeletonize_image(img, roads)
    
    if not DEBUG_MODE:
        plot_images({'Imagem Original': img, 'Imagem Binarizada': thresholded_image})
        plot_images({'Imagem Contornada': contour_line_image, 'Imagem Esqueleto': skeleton_image})
    else:
        # Crie uma cópia das variáveis locais que são imagens antes de iterar
        local_vars = dict(locals())  # Faz uma cópia das variáveis locais
        debug_variables = {key: local_vars[key] for key in local_vars if key != 'kernel' and isinstance(local_vars[key], np.ndarray)}
        
        debug_items = list(debug_variables.items())  # Convertemos o dicionário em uma lista de pares (nome, variável)
        
        for i in range(len(debug_items) - 1):
            # Criando um dicionário com o par atual e o próximo par
            images_to_plot = {
                debug_items[i][0]: debug_items[i][1],     # Primeiro item do par
                debug_items[i + 1][0]: debug_items[i + 1][1]  # Segundo item do par
            }
            plot_images(images_to_plot)
        
# Função principal
if __name__ == '__main__':
    main()