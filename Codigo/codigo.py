import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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

def main():
    # Carregar imagem
    img_path = '../Imagens/imagem1.png'

    # Carregar imagem em RGB (OpenCV lê em BGR por padrão)
    img = cv.imread(img_path, cv.IMREAD_COLOR)

    #carregar threshold test
    img2 = cv.imread('../Imagens/threshold1.png', cv.IMREAD_COLOR)

    # Converter imagem para escala de cinza
    gray = convert_to_gray(img)

    #tophat 
    kernel = np.ones((51,51), np.uint8)
    white_tophat = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)

    #OTSU threshold
    #threshold_number, gray_image_threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #threshold_number = 110
    threshold_number = 120
    _, gray_image_threshold2 = cv.threshold(gray, threshold_number, 255, cv.THRESH_BINARY)


    images = {
        #'Original Image': img,
        'Gray Image': gray,
        'tophat': white_tophat,
        'Threshold Artigo': img2,
        #'Threshold Image': gray_image_threshold,
        'Threshold Image 2': gray_image_threshold2,
    }

    # Plotar ambas as imagens
    plot_images(images)

if __name__ == '__main__':
    main()
