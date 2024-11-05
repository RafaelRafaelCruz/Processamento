import cv2
import numpy
import matplotlib.pyplot as plt

imagem = cv2.imread('entrada.jpg')

print(f"Largura em pixels {imagem.shape[1]}")
print(f"Altura em pixels {imagem.shape[0]}")

cv2.imshow("Imagem", imagem)
cv2.waitKey(0)

(b, g, r) = imagem[0, 0]
print(b, g, r)


canalBlue = numpy.zeros((imagem.shape[0], imagem.shape[1], imagem.shape[2]), dtype=numpy.uint8)
canalGreen = numpy.zeros((imagem.shape[0], imagem.shape[1], imagem.shape[2]), dtype=numpy.uint8)
canalRed = numpy.zeros((imagem.shape[0], imagem.shape[1], imagem.shape[2]), dtype=numpy.uint8)

canalBlue[:, :, 0] = imagem[:, :, 0]
canalGreen[:, :, 1] = imagem[:, :, 1]
canalRed[:, :, 2] = imagem[:, :, 2]

canalGrey = numpy.zeros((imagem.shape[0], imagem.shape[1]), dtype=numpy.uint8)
row, col = imagem.shape[0:2]

for i in range(row):
    for j in range(col):
        # // transforma a divisão em um número inteiro
        canalGrey[i, j] = sum(imagem[i, j]) // 3
        
cv2.imshow("Canal Grey", canalGrey)

cv2.imshow("Canal Blue", imagem[:, :, 0])
cv2.imshow("Canal Green", imagem[:, :, 1])
cv2.imshow("Canal Red", imagem[:, :, 2])

cv2.imshow("Canal Blue", canalBlue)
cv2.imshow("Canal Green", canalGreen)
cv2.imshow("Canal Red", canalRed)
cv2.waitKey(0)

cv2.imwrite("saida.jpg", imagem)
