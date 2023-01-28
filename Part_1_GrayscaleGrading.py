import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

# Read image
img = cv2.imread("nt1.jpg",0)
img = cv2.resize(img,(300,500))



# image normalization

def normalize(img):
    img = (img - np.min(img))/ (np.max(img) - np.min(img)) * 255.0
    return img


r_left, r_right = 150, 230
r_min, r_max = 0, 255
level_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if r_left <= img[i, j] <= r_right:
            level_img[i, j] = r_max
        else:
            level_img[i, j] = img[i, j]

cv2.imshow('origin image', img)
cv2.imshow('level image', level_img)
cv2.waitKey(0)