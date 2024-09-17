# Projeto de Segmentação de Rodovias com Processamento de Imagens

Este projeto implementa uma série de técnicas de processamento de imagens para segmentar rodovias em imagens de satélite utilizando as bibliotecas OpenCV, Numpy, Matplotlib e Skimage.

## Dependências
O projeto utiliza as seguintes bibliotecas Python:

OpenCV (cv2)
Numpy (numpy)
Matplotlib (matplotlib)
Skimage (skimage)

Para instalar as dependências, utilize o seguinte comando:
```bash
pip install opencv-python numpy matplotlib scikit-image
```

## Descrição
O script realiza o processamento de imagens com os seguintes passos:

Binarização da Imagem: A imagem é convertida para o espaço de cores HSV, e os canais H e S são binarizados usando limiarização de Otsu e threshold fixo.
Remoção de Pequenos Componentes: Componentes conectados muito pequenos são removidos da imagem binarizada para reduzir ruídos.
Conexão de Componentes à Borda: Apenas os componentes conectados às bordas da imagem são mantidos, pois são mais prováveis de fazerem parte de uma rodovia.
Preenchimento de Buracos Pequenos: Pequenos buracos nas áreas detectadas são fechados.
Detecção de Rodovias: A partir das formas geométricas dos componentes conectados, são detectadas as rodovias presentes na imagem.
Esqueletização: Após a detecção, é gerado o esqueleto das rodovias para facilitar a análise de sua estrutura.

## Estrutura do Código

O código está dividido em várias funções para modularidade:

plot_images(): Exibe imagens lado a lado para análise visual.
invert_channel(): Inverte os valores de um canal de imagem.
binarize_image(): Realiza a binarização da imagem nos canais H e S.
remove_small_components(): Remove componentes com área inferior a um valor mínimo.
connect_to_border(): Mantém apenas os componentes conectados à borda da imagem.
close_small_holes(): Fecha buracos pequenos na imagem binarizada.
detect_road(): Detecta rodovias com base na relação área/perímetro dos componentes.
road_contour_line(): Gera uma imagem com as bordas detectadas das rodovias.
skeletonize_image(): Esqueletiza as rodovias detectadas.

## Execução
Para executar o script, basta rodar o seguinte comando no terminal:
```bash
python <nome_do_script>.py
```

## Parâmetros e Ajustes
DEBUG_MODE: Quando DEBUG_MODE está ativado (True), o código exibe uma sequência detalhada de passos intermediários do processamento. Com DEBUG_MODE desativado (False), o código exibe as imagens finais do processamento.
SMALL_COMPONENT_MIN_AREA: Define o valor mínimo de área para componentes conectados a serem considerados durante o processo de remoção de pequenos componentes. Este valor pode ser ajustado dependendo das características da imagem.

## Exemplo de Saída
Ao executar o script, as seguintes imagens são geradas e exibidas:

Imagem Original e Imagem Binarizada.
Imagem das Rodovias Extraídas e Imagem de Contorno.
Imagem Esqueletizada da Rodovia.