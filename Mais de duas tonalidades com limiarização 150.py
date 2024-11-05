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

# Calcula e exibe o histograma em preto e branco
plt.hist(canalGrey.ravel(), 256, [0, 256], color='black')
plt.title("Histograma da Imagem em Escala de Cinza")
plt.xlabel("Intensidade de Cinza")
plt.ylabel("Número de Pixels")
plt.show()

# Aplicação do threshold com ponto de corte de 150
_, imagem_threshold = cv2.threshold(canalGrey, 150, 255, cv2.THRESH_BINARY)

# Exibe a imagem com threshold
cv2.imshow("Imagem com threshold (valor 150)", imagem_threshold)
cv2.waitKey(0)

# Salva a imagem resultante
cv2.imwrite("imagem_threshold.jpg", imagem_threshold)

# Limiarização multinível
# Definindo múltiplos limites
limites = [50, 100, 150]
valores = [0, 127, 255, 255]

# Criando uma nova imagem para armazenar a saída da limiarização multinível
imagem_multilevel = np.zeros_like(canalGrey)

# Aplicando a limiarização multinível
for i in range(len(limites) - 1):
    mask = (canalGrey >= limites[i]) & (canalGrey < limites[i + 1])
    imagem_multilevel[mask] = valores[i]

# Exibe a imagem com limiarização multinível
cv2.imshow("Imagem com Limiarização Multinível", imagem_multilevel)
cv2.waitKey(0)

# Salva a imagem resultante
cv2.imwrite("imagem_multilevel_threshold.jpg", imagem_multilevel)

# Finaliza todas as janelas
cv2.destroyAllWindows()
