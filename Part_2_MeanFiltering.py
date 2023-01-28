import numpy as np
import cv2

# Read image
img = cv2.imread("picture2.jpg")
img = cv2.resize(img,(300,500))
h, w = img.shape[0:2]


# image normalization

def normalize(img):
    img = (img - np.min(img))/ (np.max(img) - np.min(img)) * 255.0
    return img

def meanFiltering1(img,size): #Size is the size of the mean filter,>=3, must be odd
    # Enter the size of the image to be filled
    num = int((size - 1) / 2)
    # Expand the incoming image with the size of num
    img = cv2.copyMakeBorder(img, num, num, num, num, cv2.BORDER_REPLICATE)
    h1, w1 = img.shape[0:2]


    img1 = np.zeros((h1, w1, 3), dtype="uint8")

    for i in range(num, h1-num):
        for j in range(num, w1-num):
            sum=0
            sum1=0
            sum2=0
            # Calculate the average value of pixels in the size * size area around the center pixel
            for k in range(i-num,i+num+1):
                for l in range(j-num,j+num+1):
                    sum=sum+img[k,l][0]
                    sum1=sum1+img[k,l][1]
                    sum2=sum2+img[k,l][2]
            # Divided by the square of the core size
            sum=sum/(size**2)
            sum1 = sum1/(size**2)
            sum2 = sum2/(size**2)
            img1[i, j]=[sum,sum1,sum2]
    img1=img1[(0+num):(h1-num),(0+num):(h1-num)]
    return img1


result=meanFiltering1(img,5)
cv2.imshow("Noise",img)
cv2.imshow("mean1result",result)
cv2.imwrite("mean2result.jpg",result)
cv2.waitKey(0)