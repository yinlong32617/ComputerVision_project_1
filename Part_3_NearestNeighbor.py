import numpy as np
import cv2


# Read image
img = cv2.imread("nature1.jpg")



# image normalization

def normalize(img):
    img = (img - np.min(img))/ (np.max(img) - np.min(img)) * 255.0
    return img

src_h = img.shape[0]
src_w = img.shape[1]
# Set picture zoom factor
dst_h = int(3 * src_h)
dst_w = int(3 * src_w)


# dst_h is the height of the target image, and dst_w is the width of the target image
def NN_interpolation(img,dst_h,dst_w):
    retimg=np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    for i in range(dst_h-1):
        for j in range(dst_w-1):
            scrx=round(i*(src_h/dst_h))
            scry=round(j*(src_w/dst_w))
            retimg[i,j]=img[scrx,scry]
    return retimg

retimg= NN_interpolation(img,dst_h,dst_w)

cv2.imwrite("nature1afterNN3.jpg", retimg)
cv2.imshow("result", retimg)
cv2.waitKey(0)

