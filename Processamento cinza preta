import cv2
import numpy as np

# Carregar a imagem colorida
imagem = cv2.imread('entrada.jpg')

# Exibir informações da imagem
print(f"Largura em pixels: {imagem.shape[1]}")
print(f"Altura em pixels: {imagem.shape[0]}")

# Criar a imagem cinza
altura, largura = imagem.shape[:2]
imagem_cinza = np.zeros((altura, largura), dtype=np.uint8)

# Converter cada pixel para cinza
for i in range(altura):
    for j in range(largura):
        # Média dos canais de cor para obter o tom de cinza
        imagem_cinza[i, j] = (int(imagem[i, j][0]) + int(imagem[i, j][1]) + int(imagem[i, j][2])) // 3

# Criar uma imagem preta com a mesma dimensão da imagem cinza
imagem_preta = np.zeros((altura, largura), dtype=np.uint8)

# Exibir a imagem original, a imagem em cinza e a imagem preta
cv2.imshow("Imagem Original", imagem)
cv2.imshow("Imagem Cinza", imagem_cinza)
cv2.imshow("Imagem Preta", imagem_preta)

# Salvar a imagem preta
cv2.imwrite("saida_preta.jpg", imagem_preta)

cv2.waitKey(0)
cv2.destroyAllWindows()
