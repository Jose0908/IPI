import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem = cv.imread('../Imagens/imagem-teste5.png')

# Transformar a imagem em YUV
imagem_yuv = cv.cvtColor(imagem, cv.COLOR_BGR2YUV)

# Separar os canais de cor
Y = imagem_yuv[:, :, 0]
U = imagem_yuv[:, :, 1]
V = imagem_yuv[:, :, 2]

# Utilizar OTSU no canal U para binarizar a imagem
_, U = cv.threshold(U, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# Utilizar OTSU no canal V para binarizar a imagem
_, V = cv.threshold(V, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# U AND V
UV = cv.bitwise_and(U, V)
# Mostrar imagem UV
plt.figure(figsize=(10, 10))
plt.imshow(UV, cmap='gray')
plt.title('U AND V')
plt.show()


#Mostrar os canais separados
plt.figure(figsize=(10, 10))
plt.subplot(131)
plt.imshow(Y, cmap='gray')
plt.title('Y')
plt.subplot(132)
plt.imshow(U, cmap='gray')
plt.title('U')
plt.subplot(133)
plt.imshow(V, cmap='gray')
plt.title('V')

plt.show()

for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        U = imagem_yuv[i, j, 1]
        V = imagem_yuv[i, j, 2]
        # Se U e V estiverem próximos a 128, então a cor é cinza
        if (U >= 120 and U <= 136) and (V >= 120 and V <= 136):
            imagem_yuv[i, j, 0] = 230

# Converter a imagem de volta para RGB
imagem = cv.cvtColor(imagem_yuv, cv.COLOR_YUV2RGB)

# Mostrar a imagem
plt.figure(figsize=(10, 10))
plt.imshow(imagem)
plt.show()




