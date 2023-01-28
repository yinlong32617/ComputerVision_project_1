import numpy as np
import cv2

# Read image into Greyscale
img = cv2.imread("picture2.jpg")
img = cv2.resize(img,(300,500))



# image normalization

def normalize(img):
    img = (img - np.min(img))/ (np.max(img) - np.min(img)) * 255.0
    return img

def MedianFilter(img,K_size=3):
    h,w,c = img.shape
        # zero padding
    pad = K_size//2
    result = np.zeros((h + 2*pad,w + 2*pad,c),dtype=np.float)
    result[pad:pad+h,pad:pad+w] = img.copy().astype(np.float)
        # Convolution
    tmp = result.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                result[pad+y,pad+x,ci] = np.median(tmp[y:y+K_size,x:x+K_size,ci])
    result = result[pad:pad+h,pad:pad+w].astype(np.uint8)
    return result

result = MedianFilter(img,11)
cv2.imshow("result", result)
cv2.imwrite("mideum2result.jpg", result)
cv2.waitKey(0)
