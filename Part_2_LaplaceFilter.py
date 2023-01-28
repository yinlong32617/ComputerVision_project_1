import numpy as np
import cv2

# Read image into Greyscale
img = cv2.imread("picture4.jpg",0)
img = cv2.resize(img,(300,500))



# image normalization

def normalize(img):
    img = (img - np.min(img))/ (np.max(img) - np.min(img)) * 255.0
    return img

#Laplace Filter
laplace = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

# Create a new image and fill the image edge
img2 = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
high, wide = img2.shape

# Copy old image to new image
for i in range(1, high - 1):
    for j in range(1, wide - 1):
        img2[i, j] = img[i - 1, j - 1]
cv2.imshow("image", img2)

# Create result image
result = np.zeros((high, wide), np.uint8)


# convolution
for i in range(0, high - 2):
    for j in range(0, wide - 2):
        sum = 0
        for row_i in range(3):
            for col_j in range(3):
                sum = laplace[row_i, col_j] * img2[i + row_i, j + col_j] + sum
        # Add an absolute value to prevent pixels from becoming negative due to convolution.
        result[i + 1, j + 1] = abs(sum)

cv2.imshow("result", result)
cv2.imwrite("Laplace4result.jpg", result)
cv2.waitKey(0)
