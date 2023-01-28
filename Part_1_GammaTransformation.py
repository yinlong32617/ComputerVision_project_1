import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

# Read image
img = cv2.imread("gt2.jpg")
img = cv2.resize(img,(300,500))

# image normalization

def normalize(img):
    img = (img - np.min(img))/ (np.max(img) - np.min(img)) * 255.0
    return img

#Gamma Transformaiton
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("gt_Graybefore.png")
plt.show()

# Take img to the power of 0.2
img1 = np.power(img/float(np.max(img)), 1.9)

plt.hist(img1.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("gt_Grayafter.png")
plt.show()

cv2.imshow('src',img)
cv2.imshow('gamma=1/1.5',img1)
cv2.imwrite("gt2_GammaTransformation.jpg", img1)
cv2.waitKey(0)