import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
imagem = cv2.imread('entrada.jpg', cv2.IMREAD_COLOR)

# Mostrar dimensões da imagem
print(f"Largura em pixels: {imagem.shape[1]}")
print(f"Altura em pixels: {imagem.shape[0]}")

# Função para alargamento de contraste com 3 trechos lineares
def alargamento_contraste(img):
    r_max = 255  # valor máximo de intensidade de entrada
    s_max = 255  # valor máximo de intensidade de saída
    
    # Criar uma LUT (Look-Up Table) com 256 valores
    lut = np.zeros(256)

    for r in range(256):
        if r <= r_max / 3:
            # Primeiro trecho linear: mapeia de 0 a 85
            lut[r] = (r / (r_max / 3)) * (s_max / 3)
        elif r <= (2 * r_max) / 3:
            # Segundo trecho linear: mapeia de 85 a 170
            lut[r] = ((r - r_max / 3) * (s_max / 3) / (r_max / 3)) + s_max / 3
        else:
            # Terceiro trecho linear: mapeia de 170 a 255
            lut[r] = ((r - (2 * r_max) / 3) * (s_max / 3) / (r_max / 3)) + (2 * s_max) / 3

    # Aplicar a LUT em cada canal da imagem
    img_resultado = cv2.LUT(img, lut.astype(np.uint8))
    return img_resultado

# Aplicar a função de alargamento de contraste
imagem_alargada = alargamento_contraste(imagem)

# Função para plotar histogramas e curvas de tom
def plot_histogram_curve(img_original, img_alargada):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Histograma da imagem original
    axs[0, 0].hist(img_original.ravel(), bins=256, range=(0, 256), color='gray')
    axs[0, 0].set_title("Histograma Original")

    # Histograma da imagem alargada
    axs[0, 1].hist(img_alargada.ravel(), bins=256, range=(0, 256), color='gray')
    axs[0, 1].set_title("Histograma Alargada")

    # Curva de tom original
    axs[1, 0].plot(np.arange(256), np.arange(256), 'gray')
    axs[1, 0].set_title("Curva de Tom Original")

    # Curva de tom após alargamento
    x = [0, 85, 170, 255]
    y = [0, s_max / 3, (2 * s_max) / 3, s_max]
    axs[1, 1].plot(np.arange(256), np.interp(range(256), x, y), 'gray')
    axs[1, 1].set_title("Curva de Tom Alargada")

    plt.tight_layout()
    plt.show()

# Exibir resultados
cv2.imshow("Imagem Original", imagem)
cv2.imshow("Imagem com Alargamento de Contraste", imagem_alargada)
plot_histogram_curve(imagem, imagem_alargada)

cv2.waitKey(0)
cv2.destroyAllWindows()
