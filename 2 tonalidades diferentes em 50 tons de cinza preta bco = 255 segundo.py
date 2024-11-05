import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem = cv2.imread('entrada.jpg')

# Exibir as dimensões da imagem
print(f"Largura em pixels: {imagem.shape[1]}")
print(f"Altura em pixels: {imagem.shape[0]}")

# Converter a imagem para tons de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Reduzir para 50 tons de cinza
imagem_50tons = (imagem_cinza // 5) * 5

# Exibir o histograma da imagem em 50 tons de cinza
plt.hist(imagem_50tons.ravel(), bins=50, range=[0, 255])
plt.title("Histograma da imagem em 50 tons de cinza")
plt.xlabel("Intensidade")
plt.ylabel("Frequência")
plt.show()

# Aplicar thresholding para separar os objetos com corte em 116
_, imagem_thresh_esquerda = cv2.threshold(imagem_50tons, 116, 255, cv2.THRESH_BINARY_INV)
_, imagem_thresh_direita = cv2.threshold(imagem_50tons, 116, 255, cv2.THRESH_BINARY)

# Exibir a imagem com somente os tons à esquerda do ponto de corte (116)
cv2.imshow("Objetos à esquerda do corte (Preto até 116)", imagem_thresh_esquerda)

# Exibir a imagem com somente os tons à direita do ponto de corte (116)
cv2.imshow("Objetos à direita do corte (Branco a partir de 116)", imagem_thresh_direita)

# Salvar as imagens processadas
cv2.imwrite("saida_50tons.jpg", imagem_50tons)
cv2.imwrite("saida_thresh_esquerda.jpg", imagem_thresh_esquerda)
cv2.imwrite("saida_thresh_direita.jpg", imagem_thresh_direita)

# Aguardar o usuário pressionar uma tecla e fechar todas as janelas
cv2.waitKey(0)
cv2.destroyAllWindows()
