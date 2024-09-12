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

def look_around(near_pixels):
    # near_pixels é uma lista, se tiver algum pixel branco, retorna True
    for pixel in near_pixels:
        if pixel == 255:
            return 255
    return 0

def and_pixels(pixel1, pixel2):
    # Função para fazer a operação AND entre dois pixels
    if pixel1 == 255 and pixel2 == 255:
        return 255
    return 0

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

def main():
    # Carregar imagem
    img_path = '../Imagens/imagem-original.png'

    # Carregar imagem em RGB (OpenCV lê em BGR por padrão)
    img = cv.imread(img_path, cv.IMREAD_COLOR)

    # Carregar threshold da imagem do artigo
    image_threshold_print = cv.imread('../Imagens/threshold-print.png', cv.IMREAD_COLOR)

    # Carregar a imagem print do artigo
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
    min_area = 100  # Ajuste o valor conforme necessário
    cleaned_mask = remove_small_components(thresholded_print, min_area)
    cleaned_mask2 = remove_small_components(thresholded_nossa, min_area)

    # Conectar a borda
    road_mask = connect_to_border(cleaned_mask)
    road_mask2 = connect_to_border(cleaned_mask2)

    # Mostrar a imagem original e a máscara resultante
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Imagem Original')
    plt.imshow(thresholded_print, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Imagem Limpa (Sem Pequenos Componentes)')
    plt.imshow(cleaned_mask, cmap='gray')

    plt.show()

    # Mostrar a imagem original e a máscara resultante
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Imagem Original Nossa')
    plt.imshow(thresholded_nossa, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Imagem Limpa (Sem Pequenos Componentes) Nossa')
    plt.imshow(cleaned_mask2, cmap='gray')

    plt.show()

    # Operação de expansão com vizinhança de oito pixels (dilatação com kernel 3x3)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv.dilate(road_mask, kernel, iterations=1)

    # Mostrar as imagens
    images = {
        #'Imagem Tons de Cinza Nossa': gray_nossa,
        #'Threshold Imagem Nossa': thresholded_nossa,
        'Threshold Artigo Print': thresholded_print,
        #'Imagem Sem Ruído Nossa': road_mask_nossa,
        'Imagem Sem Ruído Artigo Print:': removed_noise_print,
        'Imagem Sem Ruído Artigo': road_mask,
        #'Imagem Sem Ruído Artigo Depois da Dilatação': road_mask2
    }
    plot_images(images)

# Função principal
if __name__ == '__main__':
    main()
