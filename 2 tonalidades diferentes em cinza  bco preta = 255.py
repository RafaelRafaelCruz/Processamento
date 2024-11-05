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

# Cálculo da média dos canais de cor
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
# Limiarização para separar objetos
imagem_threshold = np.zeros_like(canalGrey)

for i in range(row):
    for j in range(col):
        if canalGrey[i, j] > 116:
            imagem_threshold[i, j] = 255  # Branca
        else:
            imagem_threshold[i, j] = 0    # Preta

# Exibe a imagem com threshold aplicado
cv2.imshow("Imagem com threshold", imagem_threshold)
cv2.waitKey(0)

# Salva a imagem resultante
cv2.imwrite("imagem_threshold.jpg", imagem_threshold)

# Finaliza todas as janelas
cv2.destroyAllWindows()
