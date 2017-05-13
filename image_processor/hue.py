import cv2
import numpy as np
from numpy import linalg as LA
import math
import copy
from scene.CirlceObject import CircleObject

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


def smoothing_filter(image, size):

    kernel = smoothingkernel(size)
    dst = cv2.filter2D(image, -1, kernel)

    return dst


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
        kh = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float)
        kv = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=np.float)
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
    maxval = np.max(g)
    #g *= 255.0 / np.max(g)

    #g = g.astype(np.int8)

    return g, gy, gx, maxval


def quadriatic_mean_circle(xCentar, yCentar, radius, picture):  # mora da se prosledi gray_scale image
    # ove 2 promenjive su za rucno izracunavanje, svuda odkomentarisati ako treba to
    sum = 0
    numberOfPixels = 0
    radius = radius * radius
    image = copy.deepcopy(picture)
    array = []
    width = np.size(picture, 0)
    height = np.size(picture, 1)
    for x in range(xCentar - radius, xCentar + 1):
        for y in range(yCentar - radius, yCentar + 1):
            if ((x - xCentar) * (x - xCentar) + (y - yCentar) * (y - yCentar) <= radius):  # radius * radius
                xSym = xCentar - (x - xCentar)
                ySym = yCentar - (y - yCentar)
                # (x, y), (x, ySym), (xSym , y), (xSym, ySym) are in the circle
                # Ovo je za slucaj bez uslova, ako zatreba brzina izvrsavanja
                # -----------------------------------------------------
                # sum += int(picture[x, y]) * int(picture[x, y])
                # sum += int(picture[x, ySym]) * int(picture[x, ySym])
                # sum += int(picture[xSym, y]) * int(picture[xSym, y])
                # sum += int(picture[xSym, ySym]) * int(picture[xSym, ySym])
                # numberOfPixels += 4
                # -----------------------------------------------------
                if ((x >= 0) & (width > x) & (y >= 0) & (height > y)):
                    image[x, y] = 255
                    array.append(int(picture[x, y]))
                    sum += int(picture[x, y]) * int(picture[x, y])
                    numberOfPixels += 1
                if ((x >= 0) & (width > x) & (ySym >= 0) & (height > ySym) & (y != ySym)):
                    image[x, ySym] = 255
                    array.append(int(picture[x, ySym]))
                    sum += int(picture[x, ySym]) * int(picture[x, ySym])
                    numberOfPixels += 1
                if ((xSym >= 0) & (width > xSym) & (y >= 0) & (height > y) & (xSym != x)):
                    image[xSym, y] = 255
                    array.append(int(picture[xSym, y]))
                    sum += int(picture[xSym, y]) * int(picture[xSym, y])
                    numberOfPixels += 1
                if ((xSym >= 0) & (width > xSym) & (ySym >= 0) & (height > ySym) & (xSym != x) & (y != ySym)):
                    image[xSym, ySym] = 255
                    array.append(int(picture[xSym, ySym]))
                    sum += int(picture[xSym, ySym]) * int(picture[xSym, ySym])
                    numberOfPixels += 1

    # pom2 = int(np.mean(array, dtype=np.float64))
    #cv2.imwrite('krugTest.png', image)
    if numberOfPixels > 0:
        pom = int(np.sqrt(sum / numberOfPixels))  # ako treba da se vrati rucno izracunavanje
    else:
        pom = 0
    return pom


def process_color_image(image):

    colorImage = image
    originalImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    returnData = list()

    img = smoothing_filter(originalImg, 4)

    img, sobelv, sobelh, maxval = sobel_filter(img, 3)

    cv2.imwrite("sobel.png", img)
    img = cv2.imread("sobel.png", 0)

    circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 10,
                               param1=10, param2=30, minRadius=10, maxRadius=20)


    if circles is None:

        return None, colorImage

    for i in circles[0, :]:

        data = [0, 0, 0, 0]
        mean = quadriatic_mean_circle(int(i[1]), int(i[0]), 5, originalImg)
        if mean <= 125:
            # print "Not a cap"
            #draw on cimg green or return data (x, y, r, flag)
            data[0] = i[0]
            data[1] = i[1]
            data[2] = i[2]
            data[3] = 1
            #cv2.circle(colorImage, (i[0], i[1]), i[2], (0, 255, 0),3)
        else:
            # print "Prob a cap"
            data[0] = i[0]
            data[1] = i[1]
            data[2] = i[2]
            data[3] = 0
            #cv2.circle(colorImage, (i[0], i[1]), i[2], (0, 0, 255),3)

        circleData = None

        if data[3] == 0:
            circleData = CircleObject("Bad")
        else:
            circleData = CircleObject("Good")

        circleData.x = data[0]
        circleData.y = data[1]
        circleData.circle_radius = data[2]

        returnData.append(circleData)

    return returnData, colorImage

def eigenvector(np_array):
    gray = cv2.cvtColor(np_array, cv2.COLOR_RGB2GRAY)
    w, v = LA.eig(gray)

    return w, v

def ROI(width, height, center_x, center_y, image):
    widthHalf = width / 2
    heightHalf = height / 2
    roi = image[center_x - widthHalf:center_x + widthHalf, center_y - heightHalf: center_y + heightHalf]
    roi = np.divide(roi, 1000.)
    return roi


