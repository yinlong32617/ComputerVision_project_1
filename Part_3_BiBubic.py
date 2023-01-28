import numpy as np
import cv2
import math


# Read image
img = cv2.imread("Synthetic2.jpg")



# image normalization

def normalize(img):
    img = (img - np.min(img))/ (np.max(img) - np.min(img)) * 255.0
    return img


src_h = img.shape[0]
src_w = img.shape[1]
# Set picture zoom factor
dst_h = int(2 * src_h)
dst_w = int(2 * src_w)

# Generate different weights of 16 pixels
def BiBubic(x):
    x=abs(x)
    if x<=1:
        return 1-2*(x**2)+(x**3)
    elif x<2:
        return 4-8*x+5*(x**2)-(x**3)
    else:
        return 0

# dst_h is the height of the target image, and dst_w is the width of the target image
def BiCubic_interpolation(img,dst_h,dst_w):
    scrH,scrW,_=img.shape
    retimg=np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    for i in range(dst_h):
        for j in range(dst_w):
            scrx=i*(src_h/dst_h)
            scry=j*(src_w/dst_w)
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            tmp=0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if x+ii<0 or y+jj<0 or x+ii>=src_h or y+jj>=src_w:
                        continue
                    tmp+=img[x+ii,y+jj]*BiBubic(ii-u)*BiBubic(jj-v)
            retimg[i,j]=np.clip(tmp,0,255)
    return retimg

retimg= BiCubic_interpolation(img,dst_h,dst_w)

cv2.imwrite("Synthetic2afterBibubic2.jpg", retimg)
cv2.imshow("result", retimg)
cv2.waitKey(0)
