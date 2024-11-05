import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar imagem
imagem = cv2.imread('entrada.jpg', cv2.IMREAD_GRAYSCALE)

# Mostrar largura e altura da imagem
print(f"Largura em pixels: {imagem.shape[1]}")
print(f"Altura em pixels: {imagem.shape[0]}")

# Transformar em 50 tons de cinza
imagem_50tons = (imagem // 5) * 5  # Reduz a intensidade para 50 tons

# Mostrar a imagem com 50 tons de cinza
cv2.imshow("Imagem 50 Tons de Cinza", imagem_50tons)

# Calcular e exibir histograma
hist = cv2.calcHist([imagem_50tons], [0], None, [256], [0, 256])
plt.figure()
plt.title("Histograma")
plt.xlabel("Intensidade de Pixel")
plt.ylabel("Número de Pixels")
plt.plot(hist)
plt.show()

# Aplicar limiarização para separar objetos na imagem (Threshold 116)
_, img_threshold = cv2.threshold(imagem_50tons, 116, 255, cv2.THRESH_BINARY)

# Mostrar imagem com threshold aplicado
cv2.imshow("Imagem Threshold", img_threshold)

# Realçar cor preta à esquerda do ponto de corte (transformar tudo à direita de 116 para 255)
imagem_preta = imagem_50tons.copy()
imagem_preta[imagem_preta > 116] = 255
cv2.imshow("Imagem Realce Preto", imagem_preta)

# Realçar cor branca à direita do ponto de corte (transformar tudo à esquerda de 116 para 0)
imagem_branca = imagem_50tons.copy()
imagem_branca[imagem_branca <= 116] = 0
cv2.imshow("Imagem Realce Branco", imagem_branca)

# Espera até que uma tecla seja pressionada e fecha todas as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()
