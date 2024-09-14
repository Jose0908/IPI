import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def convert_to_gray(img):
    # Função para converter a imagem RGB para escala de cinza usando a fórmula dada
    gray = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gray[i, j] = 0.229 * img[i, j, 2] + 0.587 * img[i, j, 1] + 0.114 * img[i, j, 0]
    return gray

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

def connect_to_border(image):
    kernel = np.ones((2, 2), np.uint8)
    # Fazer dilatação na imagem
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

def remove_small_components(image, min_area):
    """
    Remove componentes conectados que têm área menor que `min_area`.
    """
    # Detectar componentes conectados
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(image, connectivity=8)

    # Crie uma máscara para os componentes maiores que min_area
    cleaned_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(1, num_labels):  # Ignorar o fundo (label 0)
        area = stats[i, cv.CC_STAT_AREA]
        if area >= min_area:
            cleaned_image[labels == i] = 255

    return cleaned_image

def get_biggest_component(image):
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(image, connectivity=8)
    max_area = 0
    max_label = 0
    # Looping through all the components
    for i in range(1, num_labels):  # Starting from 1 to skip the background label (0)
        area = stats[i, cv.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = i
    # Returning a binary mask where the biggest component is marked as True
    biggest_component = (labels == max_label).astype(np.uint8)
    return biggest_component * 255  # Convert to binary image (0 or 255)

def closeSmallHoles(image, min_area):
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


def main():
    img = cv.imread('../Imagens/imagem-original.png', cv.IMREAD_COLOR)
    image_threshold_print = cv.imread('../Imagens/threshold-print.png', cv.IMREAD_COLOR)
    image_removed_noise_print = cv.imread('../Imagens/removed-noise-print2.png', cv.IMREAD_COLOR)

    # Converter imagem para escala de cinza
    gray_nossa = convert_to_gray(img)
    gray_artigo = convert_to_gray(image_threshold_print)
    gray_artigo2 = convert_to_gray(image_removed_noise_print)

    # Aplicar limiarização de Otsu
    _, thresholded_nossa = cv.threshold(gray_nossa, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, thresholded_print = cv.threshold(gray_artigo, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, removed_noise_print = cv.threshold(gray_artigo2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Remover pequenos componentes conectados à borda (ajustar `min_area` conforme necessário)
    min_area = 80  # Ajuste o valor conforme necessário
    cleaned_mask = remove_small_components(thresholded_print, min_area)
    cleaned_mask2 = remove_small_components(thresholded_nossa, min_area)

    # Conectar a borda
    road_mask = connect_to_border(cleaned_mask)
    road_mask2 = connect_to_border(cleaned_mask2)

    kernel = np.ones((4, 4), np.uint8)
    dilated_image = cv.dilate(road_mask, kernel, iterations=1)

    # Fechar buracos pequenos
    min_area = 570  # Ajuste o valor conforme necessário
    dilated_image2 = closeSmallHoles(dilated_image, min_area)


    images = {
        'Antes do processamento': dilated_image,
        'Depois do processamento': dilated_image2,
    }
    plot_images(images)

    images = {
        'Imagem Original Nossa': cleaned_mask2,
        'Imagem Limpa (Sem Pequenos Componentes) Nossa': road_mask2,
    }
    #plot_images(images)

    # Aplicar uma operação de abertura para remover pequenos ruídos
    kernel = np.ones((5, 5), np.uint8)
    opening = cv.morphologyEx(dilated_image2, cv.MORPH_OPEN, kernel)

    # Encontrar os contornos das regiões brancas
    contours, _ = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Função para calcular a razão área/perímetro
    def calculate_ratio(contour):
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            return 0
        return np.sqrt(area) / perimeter

    # Filtrar os contornos que podem representar rodovias
    min_ratio = 0.1  # Ajuste esse valor conforme necessário
    road_contours = [cnt for cnt in contours if calculate_ratio(cnt) > min_ratio]

    # Criar uma imagem de saída
    output = np.zeros_like(dilated_image2)
    cv.drawContours(output, road_contours, -1, 255, thickness=cv.FILLED)

    # UMA IMAGEM MENOS A OUTRA
    output = cv.subtract(dilated_image2, output)

    # Exibir o resultado
    plt.figure(figsize=(10, 10))
    plt.imshow(output, cmap='gray')
    plt.title('Rodovias Extraídas')
    plt.show()

# Função principal
if __name__ == '__main__':
    main()