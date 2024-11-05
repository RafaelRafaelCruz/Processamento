import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem
imagem = cv2.imread('entrada.jpg')
print(f"Largura em pixels: {imagem.shape[1]}")
print(f"Altura em pixels: {imagem.shape[0]}")

# Converte a imagem para escala de cinza
canalGrey = np.zeros((imagem.shape[0], imagem.shape[1]), dtype=np.uint8)
row, col = imagem.shape[0:2]

for i in range(row):
    for j in range(col):
        canalGrey[i, j] = sum(imagem[i, j]) // 3

# Exibe a imagem em escala de cinza
cv2.imshow("Imagem em escala de cinza", canalGrey)
cv2.waitKey(0)

# Calcula e exibe o histograma
plt.hist(canalGrey.ravel(), 256, [0, 256])
plt.title("Histograma da Imagem em Escala de Cinza")
plt.xlabel("Intensidade de Cinza")
plt.ylabel("Número de Pixels")
plt.show()

# Aplicação do threshold com ponto de corte de 116
_, imagem_threshold_esq = cv2.threshold(canalGrey, 116, 255, cv2.THRESH_BINARY_INV)
_, imagem_threshold_dir = cv2.threshold(canalGrey, 116, 255, cv2.THRESH_BINARY)

# Exibe a imagem com threshold à esquerda do ponto de corte
cv2.imshow("Imagem com threshold (valores à esquerda do ponto de corte)", imagem_threshold_esq)
cv2.waitKey(0)

# Exibe a imagem com threshold à direita do ponto de corte
cv2.imshow("Imagem com threshold (valores à direita do ponto de corte)", imagem_threshold_dir)
cv2.waitKey(0)

# Salva as imagens resultantes
cv2.imwrite("imagem_threshold_esq.jpg", imagem_threshold_esq)
cv2.imwrite("imagem_threshold_dir.jpg", imagem_threshold_dir)

# Finaliza todas as janelas
cv2.destroyAllWindows()
