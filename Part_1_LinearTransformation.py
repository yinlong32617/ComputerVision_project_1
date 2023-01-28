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

A=[-0.5,1.5,0.5,1,-1]
B=[0,20,-50,255]
def LinearTransformation(a,b,img):
    for a in range(len(A)):
        for b in range(len(B)):
            aa = A[a]
            bb = B[b]
            print(aa,bb)
            if aa <0 and bb==0:
                new_img = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
                for i in range(new_img.shape[0]):
                    for j in range(new_img.shape[1]):
                        new_img[i][j] = img[i][j]*a + b
                cv2.imshow('a<0 and b=0', new_img)
                cv2.waitKey(0)

            elif aa>1:
                new_img = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
                for i in range(new_img.shape[0]):
                    for j in range(new_img.shape[1]):
                        if img[i][j] * a + b > 255:
                            new_img[i][j] = 255
                        else:
                            new_img[i][j] = img[i][j] * a + b
                cv2.imshow('a>1', new_img)
                cv2.waitKey(0)

            elif aa <1:
                new_img = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
                for i in range(new_img.shape[0]):
                    for j in range(new_img.shape[1]):
                        new_img[i][j] = img[i][j] * a + b
                cv2.imshow('a<1', new_img)
                cv2.waitKey(0)

            elif aa== -1 and bb != 0:
                new_img4 = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
                for i in range(new_img4.shape[0]):
                    for j in range(new_img4.shape[1]):
                        pix = img[i][j] * a + b
                        if pix > 255:
                            new_img4[i][j] = 255
                        elif pix < 0:
                            new_img4[i][j] = 0
                        else:
                            new_img4[i][j] = pix
                cv2.imshow('a=-1,b!=0', new_img)
                cv2.waitKey(0)

            elif aa== -1 and bb == 255:
                new_img5 = 255 - img
                cv2.imshow('a=-1,b=255', new_img)
                cv2.waitKey(0)

            else:
                break

    return new_img

result = LinearTransformation(A,B,img)
