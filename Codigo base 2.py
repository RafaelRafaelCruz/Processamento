# 0 -  Importação das bibliotecas necessárias

 

# OpenCV e Numpy

import cv2

import numpy as np

import processImgFunctions

 

# 1 - Lendo imagem

 

flu = cv2.imread('img/flu_liberta.jpg')

 

# 2 - Explorando propriedades das imagem

 

# Linhas, colunas e canais

print(f'Linhas: {flu.shape[0]}\nColunas: {flu.shape[0]}\nCanais: {flu.shape[2]}')

 

# Mostra imagem

cv2.imshow('Fluminense campeao', flu)

cv2.waitKey(0)

 

# Cores de um pixel

(b, g, r) = flu[0, 0]

 

# 3 - Separando uma imagem em cores

 

# Cria uma matriz RGB para cada canal de cor desejado

flu_blue = np.zeros((flu.shape[0], flu.shape[1], flu.shape[2]), dtype = np.uint8)

flu_green = np.zeros((flu.shape[0], flu.shape[1], flu.shape[2]), dtype = np.uint8)

flu_red = np.zeros((flu.shape[0], flu.shape[1], flu.shape[2]), dtype = np.uint8)

 

# Atribui a cada matriz RGB para seu canal correspondente, o canal da matriz da imagem original

flu_blue[:, :, 0] = flu[:, :, 0]

flu_green[:, :, 1] = flu[:, :, 1]

flu_red[:, :, 2] = flu[:, :, 2]

 

# Mostra cada canal

cv2.imshow('flu blue', flu_blue)

cv2.imshow('flu green', flu_green)

cv2.imshow('flu red', flu_red)

cv2.waitKey(0)

 

# 4 - Mostra a imagem em tons de cinza

cv2.imshow('flu cinza', processImgFunctions.to_gray(flu))

cv2.waitKey(0)

 
# Retorna uma imagem em tons de cinza

def to_gray(img):

    import numpy as np


    img_cinza = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)


    for linha in range(img.shape[0]):

        for coluna in range(img.shape[1]):

            img_cinza[linha, coluna] = img[linha, coluna].sum() // 3


    return img_cinza
