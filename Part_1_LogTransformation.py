import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

# Read image and change into Grayscale
img = cv2.imread("picture1.jpg",0)
img = cv2.resize(img,(300,500))



# image normalization

def normalize(img):
    img = (img - np.min(img))/ (np.max(img) - np.min(img)) * 255.0
    return img



# Log Argument
def log(c,img):
    img = img.astype(np.float)
    result = c*np.log(1.0+img)
    result = normalize(result)
    return result.astype(np.uint8)


result = log(4, img)


cv2.imshow("result1", result)
cv2.imwrite("picture1_log_result.jpg", result)
cv2.waitKey(0)