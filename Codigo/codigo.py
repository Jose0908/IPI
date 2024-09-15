import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from skimage.morphology import skeletonize


def inverter_canal(canal):
    # Função para inverter o canal de cor
    return abs(255 - canal)

def getGrayPixels(image):
    #convert to hsv
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    s_invertido = inverter_canal(hsv[:, :, 1])

    #mostrar hsv
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
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

    return thf

    # plt.subplot(1, 3, 1)
    # plt.imshow(th, cmap='gray', vmin=0, vmax=255)
    # plt.title('H binarizado')
    # plt.axis('off')  # Desativa os eixos

    # Segundo subplot para 'S binarizado'
    # plt.subplot(1, 3, 2)
    # plt.imshow(th2, cmap='gray', vmin=0, vmax=255)
    # plt.title('S binarizado')
    # plt.axis('off')

    # Terceiro subplot para 'And H e S binarizado'
    # plt.subplot(1, 3, 3)
    # plt.imshow(thf, cmap='gray', vmin=0, vmax=255)
    # plt.title('And H e S binarizado')
    # plt.axis('off')
    # plt.show()

def brightGray(image):
    # Converter a imagem para float32 para operações aritméticas precisas
    image = image.astype(np.float32)
    
    # Separar os canais de cor (B, G, R)
    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]
    
    # Criar máscaras baseadas nas condições
    mask_brilho = (np.abs(R - G) < 25) & (np.abs(R - B) < 25) & (np.abs(G - B) < 25)
    mask_cor = (R > 110) & (R < 200) & (G > 110) & (G < 200) & (B > 110) & (B < 200)
    
    # Combinar as máscaras
    mask = mask_brilho & mask_cor
    
    # Aplicar as operações usando as máscaras
    image[mask] += 50
    image[~mask] /= 2
    
    # Garantir que os valores estejam dentro do intervalo [0, 255]
    image = np.clip(image, 0, 255)
    
    # Converter a imagem de volta para uint8
    image = image.astype(np.uint8)
    
    return image

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

def morphological_gradient(image):
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv.dilate(image, kernel, iterations=1)
    erosion = cv.erode(image, kernel, iterations=1)
    return dilation - erosion

def contour_image(gray_image, contour):
    return cv.bitwise_or(gray_image, contour)

# Função para calcular a razão área/perímetro
def calculate_ratio(contour):
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return np.sqrt(area) / perimeter

def main():
    img = cv.imread('../Imagens/imagem-teste10.png', cv.IMREAD_COLOR)
    thresholded_nossa = getGrayPixels(img) 
    # Mostrar thresholded_nossa
    #plot_images({'Imagem Original': img, 'Imagem Thresholded': thresholded_nossa})

    # Remover pequenos componentes conectados à borda (ajustar `min_area` conforme necessário)
    min_area = 80  
    cleaned_mask = remove_small_components(thresholded_nossa, min_area)
    #Mostrar a imagem limpa
    #plot_images({'Imagem Original': thresholded_nossa, 'Imagem Limpa': cleaned_mask})

    # Conectar a borda
    road_mask = connect_to_border(cleaned_mask)
    #Mostrar a imagem conectada
    #plot_images({'Imagem Original': cleaned_mask, 'Imagem Conectada': road_mask})

    # Dilatar a imagem
    kernel = np.ones((3, 3), np.uint8)
    dilated_image_nossa = cv.dilate(road_mask, kernel, iterations=1)
    #Mostrar a imagem dilatada
    #plot_images({'Imagem Original': road_mask, 'Imagem Dilatada': dilated_image_nossa})

    # Fechar buracos pequenos
    #min area é o raiz do tamanho da imagem dividido por 2
    tamanho_imagem = np.sqrt(img.shape[0] * img.shape[1])
    min_area = tamanho_imagem // 2
    #min_area = 170
    dilated_image_nossa2 = closeSmallHoles(dilated_image_nossa, min_area)
    #Mostrar a imagem com buracos fechados
    #plot_images({'Imagem Original': dilated_image_nossa, 'Imagem com Buracos Fechados': dilated_image_nossa2})

    # Aplicar um fechamento
    kernel = np.ones((5, 5), np.uint8)
    opening_nossa = cv.morphologyEx(dilated_image_nossa2, cv.MORPH_CLOSE, kernel)
    #Mostrar a imagem com fechamento
    #plot_images({'Imagem Original': dilated_image_nossa2, 'Imagem com Fechamento': opening_nossa})

    # Encontrar os contornos das regiões brancas
    contours_nossa, _ = cv.findContours(opening_nossa, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    #ratios = [calculate_ratio(cnt) for cnt in contours_nossa]
    #print (ratios)
    #Mostrar os contornos
    #plot_images({'Imagem Original': opening_nossa, 'Contornos': cv.drawContours(np.zeros_like(opening_nossa), contours_nossa, -1, 255, thickness=2)})

    # Filtrar os contornos que podem representar rodovias
    min_ratio = 0.1  # Ajuste esse valor conforme necessário
    #mostrar cada um dos ratios

    road_contours_nossa = [cnt for cnt in contours_nossa if calculate_ratio(cnt) < min_ratio]
    # Mostrar road_contours_nossa 
    #plot_images({'Imagem Original': dilated_image_nossa2, 'Road Contours': cv.drawContours(np.zeros_like(dilated_image_nossa2), road_contours_nossa, -1, 255, thickness=cv.FILLED)})

    #print (f'Número de rodovias detectadas: {len(road_contours_nossa)}')

    # Criar uma imagem de saída
    output = np.zeros_like(dilated_image_nossa2)
    cv.drawContours(output, road_contours_nossa, -1, 255, thickness=cv.FILLED)

    output = cv.subtract(dilated_image_nossa2, output)
    output2 = cv.subtract(dilated_image_nossa2, output)
    plot_images({'Imagem Original': dilated_image_nossa2, 'Rodovias Extraídas': output2})

    gradient = morphological_gradient(output2)

    #transformar imagem em escala de cinza

    # Sobrepor os contornos brancos na imagem RGB original
    # O bitwise_or é utilizado para manter os contornos brancos sobre a imagem RGB
    
    #converter img para escala de cinza
    gray_nossa = convert_to_gray(img)

    final_image = cv.bitwise_or(gray_nossa, gradient)

    #converter para rgb
    final_image = cv.cvtColor(final_image, cv.COLOR_BGR2RGB)

    # Mostrar a imagem final com os contornos sobrepostos
    plt.figure(figsize=(10, 10))
    plt.imshow(final_image)
    plt.title('Rodovias Extraídas')
    plt.axis('off')  # Oculta os eixos
    plt.show()

    # Normalizar a imagem para 0 e 1 (necessário para a função skeletonize)
    output2 = output2 // 255

    # Aplicar a esqueletização
    skeleton = skeletonize(output2)

    # Converter de volta para 0-255 para exibição com o OpenCV/Matplotlib
    skeleton = (skeleton * 255).astype(np.uint8)

    # Mostrar a imagem esqueletizada
    plt.figure(figsize=(10, 10))
    plt.imshow(skeleton, cmap='gray')
    plt.title('Imagem Esqueletizada')
    plt.axis('off')  # Oculta os eixos
    plt.show()

    # jogar a imagem skeletonizada em cima da imagem original

    final_image2 = cv.bitwise_or(gray_nossa, skeleton)
    #mostrar
    final_image2 = cv.cvtColor(final_image2, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(final_image2)
    plt.title('Rodovias Extraídas')
    plt.axis('off')  # Oculta os eixos
    plt.show()

# Função principal
if __name__ == '__main__':
    main()