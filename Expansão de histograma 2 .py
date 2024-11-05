import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem em escala de cinza
imagem = cv2.imread('entrada.jpg', cv2.IMREAD_GRAYSCALE)

# Mostrar dimensões da imagem
print(f"Largura em pixels: {imagem.shape[1]}")
print(f"Altura em pixels: {imagem.shape[0]}")

# Função para equalização de histograma
def equalizar_histograma(img):
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    cdf = hist.cumsum()  # Cálculo da função de distribuição cumulativa
    cdf_normalizada = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  # Normalizar para o intervalo [0, 255]
    cdf_normalizada = cdf_normalizada.astype(np.uint8)
    
    # Mapeia a imagem original usando a CDF normalizada
    img_equalizada = cdf_normalizada[img]
    return img_equalizada

# Aplica a equalização de histograma
imagem_equalizada = equalizar_histograma(imagem)

# Função para plotar histogramas e curvas de tom
def plot_histogram_curve(img_original, img_equalizada):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Histograma da imagem original
    axs[0, 0].hist(img_original.ravel(), bins=256, range=(0, 256), color='gray')
    axs[0, 0].set_title("Histograma Original")

    # Histograma da imagem equalizada
    axs[0, 1].hist(img_equalizada.ravel(), bins=256, range=(0, 256), color='gray')
    axs[0, 1].set_title("Histograma Equalizado")

    # Curva de tom original (linear)
    axs[1, 0].plot(np.arange(256), np.arange(256), 'gray')
    axs[1, 0].set_title("Curva de Tom Original")

    # Curva de tom após equalização (CDF normalizada)
    hist, bins = np.histogram(img_original.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalizada = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    axs[1, 1].plot(np.arange(256), cdf_normalizada, 'gray')
    axs[1, 1].set_title("Curva de Tom Equalizada")

    plt.tight_layout()
    plt.show()

# Exibir resultados
cv2.imshow("Imagem Original", imagem)
cv2.imshow("Imagem Equalizada", imagem_equalizada)
plot_histogram_curve(imagem, imagem_equalizada)

cv2.waitKey(0)
cv2.destroyAllWindows()
