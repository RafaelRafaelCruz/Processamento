import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem
imagem = cv2.imread('entrada.jpg')
print(f"Largura em pixels: {imagem.shape[1]}")
print(f"Altura em pixels: {imagem.shape[0]}")

# Converte para tons de cinza
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Reduz a escala para 50 tons de cinza
imagem_gray_50 = (imagem_gray // 5) * 5

# Exibe o histograma da imagem em tons de cinza
plt.hist(imagem_gray_50.ravel(), bins=256, range=[0, 256], color='gray')
plt.title("Histograma da Imagem em 50 Tons de Cinza")
plt.xlabel("Intensidade")
plt.ylabel("Número de Pixels")
plt.show()

# Limiarização - Separação de objetos usando o ponto de corte de 116
_, imagem_thresh_preto = cv2.threshold(imagem_gray_50, 116, 255, cv2.THRESH_BINARY_INV)
_, imagem_thresh_branco = cv2.threshold(imagem_gray_50, 116, 255, cv2.THRESH_BINARY)

# Exibe as imagens com a limiarização
cv2.imshow("Imagem Original", imagem)
cv2.imshow("Imagem em 50 Tons de Cinza", imagem_gray_50)
cv2.imshow("Limiar - Preto à Esquerda de 116", imagem_thresh_preto)
cv2.imshow("Limiar - Branco à Direita de 116", imagem_thresh_branco)

cv2.waitKey(0)
cv2.destroyAllWindows()
