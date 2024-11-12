import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem em tons de cinza
imagem = cv2.imread('entrada.jpg', cv2.IMREAD_GRAYSCALE)

# Mostrar dimensões da imagem
print(f"Largura em pixels: {imagem.shape[1]}")
print(f"Altura em pixels: {imagem.shape[0]}")

# Função de equalização de histograma
def equalizar_histograma(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()  # Cálculo da CDF
    cdf_normalizada = cdf * hist.max() / cdf.max()  # Normalização da CDF
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf = np.ma.filled(cdf, 0).astype('uint8')  # Evitar valores inválidos
    return cdf[img]

# Função de alargamento de contraste com 3 trechos lineares
def alargamento_contraste(img, pontos):
    x = [p[0] for p in pontos]
    y = [p[1] for p in pontos]
    lut = np.zeros(256)

    for r in range(256):
        if r <= x[1]:  # Primeiro trecho
            lut[r] = y[0] + (r - x[0]) * ((y[1] - y[0]) / (x[1] - x[0]))
        elif r <= x[2]:  # Segundo trecho
            lut[r] = y[1] + (r - x[1]) * ((y[2] - y[1]) / (x[2] - x[1]))
        else:  # Terceiro trecho
            lut[r] = y[2] + (r - x[2]) * ((y[3] - y[2]) / (x[3] - x[2]))

    img_resultado = cv2.LUT(img, lut.astype(np.uint8))
    return img_resultado

# Equalizar histograma da imagem original
imagem_equalizada = equalizar_histograma(imagem)

# Aplicar alargamento de contraste nos pontos especificados
pontos = [(0, 0), (70, 50), (140, 200), (255, 255)]
imagem_alargada = alargamento_contraste(imagem_equalizada, pontos)

# Função para plotar histogramas e curvas de tom
def plot_histogram_curve(img_original, img_equalizada, img_alargada):
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    # Histograma da imagem original
    axs[0, 0].hist(img_original.ravel(), bins=256, range=(0, 256), color='gray')
    axs[0, 0].set_title("Histograma Original")

    # Histograma da imagem equalizada
    axs[1, 0].hist(img_equalizada.ravel(), bins=256, range=(0, 256), color='gray')
    axs[1, 0].set_title("Histograma Equalizado")

    # Histograma da imagem alargada
    axs[2, 0].hist(img_alargada.ravel(), bins=256, range=(0, 256), color='gray')
    axs[2, 0].set_title("Histograma Alargado")

    # Curva de tom original
    axs[0, 1].plot(np.arange(256), np.arange(256), 'gray')
    axs[0, 1].set_title("Curva de Tom Original")

    # Curva de tom após equalização
    axs[1, 1].plot(np.arange(256), np.arange(256), 'gray')
    axs[1, 1].set_title("Curva de Tom Equalizada")

    # Curva de tom após alargamento
    x = [p[0] for p in pontos]
    y = [p[1] for p in pontos]
    axs[2, 1].plot(np.arange(256), np.interp(range(256), x, y), 'gray')
    axs[2, 1].set_title("Curva de Tom Alargada")

    plt.tight_layout()
    plt.show()

# Exibir resultados
cv2.imshow("Imagem Original", imagem)
cv2.imshow("Imagem Equalizada", imagem_equalizada)
cv2.imshow("Imagem com Alargamento de Contraste", imagem_alargada)
plot_histogram_curve(imagem, imagem_equalizada, imagem_alargada)

cv2.waitKey(0)
cv2.destroyAllWindows()
