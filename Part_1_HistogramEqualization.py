import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

# Read image and change into Grayscale
img = cv2.imread("picture1.jpg",0)
img = cv2.resize(img,(300,500))

# histogram equalization

def hist_equal(img, z_max=255):
    H, W = img.shape
    # S is the total of pixels
    S = H * W * 1.

    result = img.copy()

    sum_h = 0.

    for i in range(1, 255):
         ind = np.where(img == i)
         sum_h += len(img[ind])
         z_result = z_max / S * sum_h
         result[ind] = z_result

    result = result.astype(np.uint8)
    return result



# histogram normalization
result = hist_equal(img)



# Display histogram
plt.hist(result.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig("picture1_histogram.png")
plt.show()

# Save result
cv2.imshow("result", result)
cv2.imwrite("picture1_histogram_result.jpg", result)
cv2.waitKey(0)














