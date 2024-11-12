import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem = cv2.imread('entrada.jpg', cv2.IMREAD_GRAYSCALE)

# Mostrar dimensões da imagem
print(f"Largura em pixels: {imagem.shape[1]}")
print(f"Altura em pixels: {imagem.shape[0]}")

# Função para alargamento de contraste com 3 trechos lineares
def alargamento_contraste(img, pontos):
    # Extrair os pontos de controle para os trechos lineares
    x = [p[0] for p in pontos]
    y = [p[1] for p in pontos]
    
    # Criar LUT com a função T(r) para cada ponto
    lut = np.zeros(256)
    
    for i in range(256):
        if i <= x[1]:  # Primeiro trecho
            lut[i] = y[0] + (i - x[0]) * ((y[1] - y[0]) / (x[1] - x[0]))
        elif i <= x[2]:  # Segundo trecho
            lut[i] = y[1] + (i - x[1]) * ((y[2] - y[1]) / (x[2] - x[1]))
        else:  # Terceiro trecho
            lut[i] = y[2] + (i - x[2]) * ((y[3] - y[2]) / (x[3] - x[2]))

    # Aplicar a LUT em cada pixel da imagem
    img_resultado = cv2.LUT(img, lut.astype(np.uint8))
    
    # Inverter a imagem para gerar o efeito de negativo
    img_negativo = cv2.bitwise_not(img_resultado)
    
    return img_negativo

# Parâmetros de alargamento: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
pontos = [(0, 0), (70, 50), (140, 200), (255, 255)]
imagem_alargada = alargamento_contraste(imagem, pontos)

# Função para plotar histogramas e curvas de tom
def plot_histogram_curve(img_original, img_alargada):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Histograma da imagem original
    axs[0, 0].hist(img_original.ravel(), bins=256, range=(0, 256), color='gray')
    axs[0, 0].set_title("Histograma Original")

    # Histograma da imagem alargada
    axs[0, 1].hist(img_alargada.ravel(), bins=256, range=(0, 256), color='gray')
    axs[0, 1].set_title("Histograma Alargada e Negativa")

    # Curva de tom original
    axs[1, 0].plot(np.arange(256), np.arange(256), 'gray')
    axs[1, 0].set_title("Curva de Tom Original")

    # Curva de tom após alargamento
    x = [p[0] for p in pontos]
    y = [p[1] for p in pontos]
    axs[1, 1].plot(np.arange(256), np.interp(range(256), x, y), 'gray')
    axs[1, 1].set_title("Curva de Tom Alargada")

    plt.tight_layout()
    plt.show()

# Exibir resultados
cv2.imshow("Imagem Original", imagem)
cv2.imshow("Imagem com Alargamento e Negativa", imagem_alargada)
plot_histogram_curve(imagem, imagem_alargada)

cv2.waitKey(0)
cv2.destroyAllWindows()
