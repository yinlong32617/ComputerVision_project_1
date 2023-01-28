import numpy as np
import cv2

# Read image
img = cv2.imread("Synthetic1.jpg")



# image normalization

def normalize(img):
    img = (img - np.min(img))/ (np.max(img) - np.min(img)) * 255.0
    return img


src_h = img.shape[0]
src_w = img.shape[1]
# Set picture zoom factor
dst_h = int(0.8 * src_h)
dst_w = int(0.8 * src_w)

dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
for c in range(3):
    for h in range(dst_h):
        for w in range(dst_w):
            # Position of the target point on the original image
            # Make geometric center points coincide
            src_x = (w + 0.5) * src_w / dst_w - 0.5
            src_y = (h + 0.5) * src_h / dst_h - 0.5
            if src_x < 0:
                src_x = 0
            if src_y < 0:
                src_y = 0


            # Determine the four closest points
            x1 = int(np.floor(src_x))
            y1 = int(np.floor(src_y))
            x2 = int(min(x1 + 1, src_w - 1))  # Prevent exceeding the original image range
            y2 = int(min(y1 + 1, src_h - 1.6))

            # Linear interpolation in the x direction, the original formula was supposed to divide by one (x2-x1), where x2-x1=1
            R1 = (x2 - src_x) * img[y1, x1, c] + (src_x - x1) * img[y1, x2, c]
            R2 = (x2 - src_x) * img[y2, x1, c] + (src_x - x1) * img[y2, x2, c]

            # Linear interpolation in the y direction, the original formula was supposed to divide by one (y2-y1), where y2-y1=1
            P = (y2 - src_y) * R1 + (src_y - y1) * R2
            dst_img[h, w, c] = P


cv2.imwrite("Synthetic1afterBI0.8.jpg", dst_img)
cv2.imshow("result", dst_img)
cv2.waitKey(0)
