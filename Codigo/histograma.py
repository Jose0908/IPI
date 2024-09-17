import cv2 as cv
import matplotlib.pyplot as plt

def show_histogram(image):
    # Verificar se a imagem é colorida (3 canais) ou em escala de cinza (1 canal)
    if len(image.shape) == 2:  # Imagem em escala de cinza
        plt.hist(image.ravel(), bins=256, range=[0, 256])
        plt.title('Histograma da Imagem em Escala de Cinza')
        plt.xlabel('Intensidade de Pixels')
        plt.ylabel('Número de Pixels')
        plt.show()
    else:  # Imagem colorida
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])
        plt.title('Histograma da Imagem Colorida')
        plt.xlabel('Intensidade de Pixels')
        plt.ylabel('Número de Pixels')
        plt.show()

# Exemplo de uso
image = cv.imread('../Imagens/imagem-teste5.png')
#transformar para escala de cinza
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
show_histogram(image)
