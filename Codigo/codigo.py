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

    # Erosão na máscara
    kernel = np.ones((1, 1), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)

    return mask

def main():
    # Carregar imagem
    img_path = '../Imagens/imagem-original.png'

    # Carregar imagem em RGB (OpenCV lê em BGR por padrão)
    img = cv.imread(img_path, cv.IMREAD_COLOR)

    # Carregar threshold da imagem do artigo
    image_threshold_artigo = cv.imread('../Imagens/threshold-print.png', cv.IMREAD_COLOR)

    # Converter imagem para escala de cinza
    gray_nossa = convert_to_gray(img)
    gray_artigo = convert_to_gray(image_threshold_artigo)

    # Aplicar limiarização de Otsu
    _, thresholded_nossa = cv.threshold(gray_nossa, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, thresholded_artigo = cv.threshold(gray_artigo, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Conectar a borda
    road_mask = connect_to_border(thresholded_artigo)
    road_mask_nossa = connect_to_border(thresholded_nossa)

    # Mostrar as imagens
    images = {
        'Imagem Tons de Cinza Nossa': gray_nossa,
        'Threshold Imagem Nossa': thresholded_nossa,
        'Threshold Artigo': thresholded_artigo,
        'Imagem Sem Ruído Nossa': road_mask_nossa,
        'Imagem Sem Ruído Artigo': road_mask
    }
    plot_images(images)

# Função principal
if __name__ == '__main__':
    main()
