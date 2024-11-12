import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem = cv2.imread('entrada.jpg', cv2.IMREAD_COLOR)

# Verificar se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar a imagem. Verifique o caminho e formato do arquivo.")
else:
    # Mostrar dimensões da imagem
    print(f"Largura em pixels: {imagem.shape[1]}")
    print(f"Altura em pixels: {imagem.shape[0]}")
