import numpy as np
# from random import randint
import cv2
import math

# import time
# from PIL import Image

def sobel_filter(im, k_size):
    im = im.astype(np.float)
    # width, height, c = im.shape
    width, height = im.shape
    c = 0
    if c > 1:
        img = 0.2126 * im[:, :, 0] + 0.7152 * im[:, :, 1] + 0.0722 * im[:, :, 2]
    else:
        img = im

    assert (k_size == 3 or k_size == 5)

    if k_size == 3:
        kh = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)
        kv = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float)
    else:
        kh = np.array([[-1, -2, 0, 2, 1],
                       [-4, -8, 0, 8, 4],
                       [-6, -12, 0, 12, 6],
                       [-4, -8, 0, 8, 4],
                       [-1, -2, 0, 2, 1]], dtype=np.float)
        kv = np.array([[1, 4, 6, 4, 1],
                       [2, 8, 12, 8, 2],
                       [0, 0, 0, 0, 0],
                       [-2, -8, -12, -8, -2],
                       [-1, -4, -6, -4, -1]], dtype=np.float)

    # gx = signal.convolve2d(img, kh, mode='same', boundary = 'symm', fillvalue=0)
    # gy = signal.convolve2d(img, kv, mode='same', boundary = 'symm', fillvalue=0)
    gx = cv2.filter2D(img, -1, kh)
    gy = cv2.filter2D(img, -1, kv)

    g = np.sqrt(gx * gx + gy * gy)
    g *= 255.0 / np.max(g)



    return g
def smoothing_filter(image,size):

    kernel = smoothingkernel(size)
    dst = cv2.filter2D(image, -1, kernel)

    return dst

def smoothingkernel(size):
    M_PI = 3.14159365359
    suma = 0
    filter = np.ones((size*2+1, size*2+1))

    square = size * size
    constant = 1 / (2 * M_PI*size*size)

    for i in range(0, size*2+1):
        for j in range(0, size*2+1):
            filter[i][j] = constant * math.exp(-0.5 * ((i - size)*(i-size) + (j - size)*(j - size)) / square)
            suma += filter[i][j]

    filter = filter / suma

    return filter

def testing():
    red = np.ones((7, 7))
    red = red / 2

    print red

def main(image):

    img = cv2.imread(image, 0)
    cv2.imshow("Regular", img)

    for i in range(1, 5):
        smoothed = smoothing_filter(img, i)
        sobelSlika3 = sobel_filter(smoothed, 3)
        sobelSlika5 =  sobel_filter(smoothed, 5)
        cv2.imwrite("gajba1_sobel3_smth_"+str(i*2+1)+".jpg", sobelSlika3)
        cv2.imwrite("gajba1_sobel5_smth_"+str(i*2+1)+".jpg", sobelSlika5)



    cv2.waitKey(0)
    cv2.destroyAllWindows()

main('gajba1.jpg')
