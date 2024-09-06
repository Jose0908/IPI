import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem RGB
imagem_rgb = cv2.imread('imagem1.png')

# Converter a imagem para tons de cinza usando a fórmula
# A fórmula é aplicada a cada pixel da imagem
imagem_cinza = (0.229 * imagem_rgb[:,:,0] + 
                0.587 * imagem_rgb[:,:,1] + 
                0.114 * imagem_rgb[:,:,2])

# Converter a imagem cinza para o formato uint8
imagem_cinza = np.uint8(imagem_cinza)

# Mostrar a imagem em cinza
plt.figure(figsize=(10, 5))
plt.imshow(imagem_cinza, cmap='gray', vmin=0, vmax=255)
plt.title('Imagem em Tons de Cinza')
plt.show()

