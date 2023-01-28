import numpy as np
import cv2

# Read image
img = cv2.imread("picture2.jpg")
img = cv2.resize(img,(300,500))



# image normalization

def normalize(img):
    img = (img - np.min(img))/ (np.max(img) - np.min(img)) * 255.0
    return img


# Gaussian filter

def GaussianFilter(img, K_size=3, sigma=1.3):
    if len(img.shape) == 3:

        H, W, C = img.shape

    else:

        img = np.expand_dims(img, axis=-1)

        H, W, C = img.shape

# Zero padding

    pad = K_size // 2

    result = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)

    result[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    # prepare Kernel

    K = np.zeros((K_size, K_size), dtype=np.float)

    for x in range(-pad, -pad + K_size):

        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

    K /= (2 * np.pi * sigma * sigma)

    K /= K.sum()

    tmp = result.copy()

    # filtering

    for y in range(H):

        for x in range(W):

            for c in range(C):
                result[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

    result = np.clip(result, 0, 255)

    result = result[pad: pad + H, pad: pad + W].astype(np.uint8)

    return result



# Gaussian Filter

result = GaussianFilter(img, 3, 1.3)

# Save result

cv2.imwrite("Gaussian2result.jpg", result)

cv2.imshow("result", result)

cv2.waitKey(0)
