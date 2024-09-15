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

def inverter_canal(canal):
    # Função para inverter o canal de cor
    return abs(255 - canal)

def getGrayPixels(image):
    #convert to hsv
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    s_invertido = inverter_canal(hsv[:, :, 1])

    #mostrar hsv
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    fig.patch.set_facecolor('#323232')  # Cor de fundo da figura (cinza escuro)
    axs[0].imshow(hsv[:, :, 0], cmap='gray', vmin=0, vmax=255)
    axs[0].set_title('H')
    axs[1].imshow(hsv[:, :, 1], cmap='gray', vmin=0, vmax=255)
    axs[1].set_title('S')
    axs[2].imshow(hsv[:, :, 2], cmap='gray', vmin=0, vmax=255)
    axs[2].set_title('V')
    plt.show()

    #utilizar OTSU no canal H
    _, th = cv.threshold(hsv[:, :, 0], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    #utilizar OTSU no canal S invertido
    S_invertido = inverter_canal(hsv[:, :, 1])
    threshold = 210
    _, th2 = cv.threshold(S_invertido, threshold, 255, cv.THRESH_BINARY)

    # fazer and entre os dois resultados
    thf = cv.bitwise_and(th, th2)

    plt.subplot(1, 3, 1)
    plt.imshow(th, cmap='gray', vmin=0, vmax=255)
    plt.title('H binarizado')
    plt.axis('off')  # Desativa os eixos

    # Segundo subplot para 'S binarizado'
    plt.subplot(1, 3, 2)
    plt.imshow(th2, cmap='gray', vmin=0, vmax=255)
    plt.title('S binarizado')
    plt.axis('off')

    # Terceiro subplot para 'And H e S binarizado'
    plt.subplot(1, 3, 3)
    plt.imshow(thf, cmap='gray', vmin=0, vmax=255)
    plt.title('And H e S binarizado')
    plt.axis('off')
    plt.show()

def main():
    img = cv.imread('../Imagens/imagem-original.png', cv.IMREAD_COLOR)
    getGrayPixels(img)

    img = cv.imread('../Imagens/imagem-teste2.png', cv.IMREAD_COLOR)
    img2 = getGrayPixels(img)

    # img = cv.imread('../Imagens/imagem-teste3.png', cv.IMREAD_COLOR)
    # getGrayPixels(img)

    img = cv.imread('../Imagens/imagem-teste4.png', cv.IMREAD_COLOR)
    getGrayPixels(img)

    img = cv.imread('../Imagens/imagem-teste5.png', cv.IMREAD_COLOR)
    getGrayPixels(img)

# Função principal
if __name__ == '__main__':
    main()
