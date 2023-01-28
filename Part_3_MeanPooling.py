import numpy as np
import cv2



# Read image
img = cv2.imread("Synthetic1.jpg",0)
b,g,r = cv2.split(img)


def pooling(img, m, n):
    a,b = img.shape
    retimg = []
    for i in range(0,a,m):
        # Record each line
        line = []
        for j in range(0,b,n):
            # Select pooled area
            x = img[i:i+m,j:j+n]
            line.append(np.sum(x)/(n*m))
        retimg.append(line)
    return np.array(retimg).astype(np.uint8)


retimg = pooling(np.array(r), 2, 2)
cv2.imwrite("Synthetic1afterMP2.jpg", retimg)
cv2.imshow("result", retimg)
cv2.waitKey(0)

