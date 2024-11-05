import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem e exibir informações básicas
imagem = cv2.imread('entrada.jpg')
print(f"Largura em pixels: {imagem.shape[1]}")
print(f"Altura em pixels: {imagem.shape[0]}")

# Converter a imagem para escala de cinza
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Definindo a função de transformação linear
def transformacao_linear(r):
    # Definir uma transformação linear simples: f(r) = ar + b
    # No caso, usaremos uma transformação de identidade onde a = 1 e b = 0
    # para mostrar a curva de tons original.
    a = 1
    b = 0
    return a * r + b

# Aplicar a transformação para cada pixel
imagem_transformada = transformacao_linear(imagem_gray)

# Exibir a imagem original e a transformada
cv2.imshow("Imagem Original em Cinza", imagem_gray)
cv2.imshow("Imagem Transformada", imagem_transformada)
cv2.waitKey(0)

# Gerar a curva de tons para 256 níveis de cinza
tons_de_cinza = np.arange(256)  # Valores de 0 a 255
curva_de_tons = transformacao_linear(tons_de_cinza)

# Plotar a curva de tons
plt.plot(tons_de_cinza, curva_de_tons, color='blue', label="f(r) = ar + b")
plt.xlabel("Intensidade Original (r)")
plt.ylabel("Intensidade Transformada (f(r))")
plt.title("Curva de Tons - Transformação Linear")
plt.legend()
plt.grid()
plt.show()

# Salvar a imagem transformada
cv2.imwrite("imagem_transformada.jpg", imagem_transformada)
