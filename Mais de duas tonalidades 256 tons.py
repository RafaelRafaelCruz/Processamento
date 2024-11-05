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

# Média simples para conversão em escala de cinza
for i in range(row):
    for j in range(col):
        canalGrey[i, j] = sum(imagem[i, j]) // 3

# Exibe a imagem em escala de cinza
cv2.imshow("Imagem em escala de cinza", canalGrey)
cv2.waitKey(0)

# Calcula e exibe o histograma em preto e branco
plt.figure(figsize=(10, 5))

# Histograma em preto
plt.subplot(1, 2, 1)
plt.hist(canalGrey.ravel(), 256, [0, 256], color='black')
plt.title("Histograma da Imagem em Escala de Cinza (Preto)")
plt.xlabel("Intensidade de Cinza")
plt.ylabel("Número de Pixels")

# Histograma em branco
plt.subplot(1, 2, 2)
plt.hist(canalGrey.ravel(), 256, [0, 256], color='white')
plt.title("Histograma da Imagem em Escala de Cinza (Branco)")
plt.xlabel("Intensidade de Cinza")
plt.ylabel("Número de Pixels")

plt.tight_layout()
plt.show()

# Limiarização multinível
# Definindo limites para a limiarização
limites = [80, 160, 240]
valores = [0, 127, 255]

# Criando uma nova imagem para a limiarização multinível
imagem_limite_multinivel = np.zeros_like(canalGrey)

for i in range(len(limites) - 1):
    mask = (canalGrey >= limites[i]) & (canalGrey < limites[i + 1])
    imagem_limite_multinivel[mask] = valores[i]

# Exibe a imagem resultante da limiarização multinível
cv2.imshow("Imagem com Limiarização Multinível", imagem_limite_multinivel)
cv2.waitKey(0)

# Salva a imagem resultante da limiarização
cv2.imwrite("imagem_limite_multinivel.jpg", imagem_limite_multinivel)

# Finaliza todas as janelas
cv2.destroyAllWindows()
