import numpy as np
from random import randint
import cv2
import math
import neuralnetwork as neural
import time
import copy
# import time
# from PIL import Image

import sys

from PySide.QtGui import QMainWindow, QPushButton, QApplication
from PySide import QtCore, QtGui

from ui_window import Ui_MainWindow

from math import hypot, pi, cos, sin
# from PIL import Image


def abs(number):
    if number > 0 or number == 0:
        return number
    else:
        return - number

def edgeEnhance(floatImg , sobelImg, width , height ,maxicanHat):

    hlimit = height - 2
    wlimit = width - 2
    secondder = cv2.filter2D(floatImg, -1, maxicanHat)

    cv2.imwrite("second_derr.png", secondder)

    for i in range(0,height):
        for j in range(0,width):
            if( i > 1 and j > 1 and i < hlimit and j < wlimit):
                if (math.fabs(secondder[j][i] + secondder[j + 1][i] + secondder[j][i + 1]) == math.fabs(secondder[j][i]) + math.fabs(secondder[j + 1][i]) + math.fabs(secondder[j][i + 1])):
                    sobelImg[j][i] = 0

            else:
                sobelImg[j][i] = 0

    return sobelImg

def hough(im, ntx=460, mry=360):

    pim = im;

    nimx, mimy = im.shape
    mry = int(mry / 2) * 2  # Make sure that this is even
#    him = Image.new("L", (ntx, mry), 255)
    phim = np.ones((nimx, mimy))
    phim = phim * 255

    rmax = hypot(nimx, mimy)
    dr = rmax / (mry / 2)
    dth = pi / ntx

    for jx in xrange(nimx):
        for iy in xrange(mimy):
            col = pim[jx][iy]
            if col == 255: continue
            for jtx in xrange(ntx):
                th = dth * jtx
                r = jx * cos(th) + iy * sin(th)
                iry = mry / 2 + int(r / dr + 0.5)
                phim[jtx][iry] -= 1
    return phim

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
def thresholdandfinddirectionmap(sobelimg, width, height, sobelh, sobelv, maxval, contthreshold):
    dirmap = np.ones((width, height))
    edgemap = np.ones((width, height))
    dirmapuchar = np.ones((width, height))
    sobel = sobelimg
    for i in range(0, height):
        for j in range(0, width):
            if(sobel[j][i] / maxval * 255 < contthreshold):
                edgemap[j][i] = 0
                dirmap[j][i] = 0
                dirmapuchar[j][i] = 0
                sobel[j][i] = 0
            else:
                edgemap[j][i] = 255
                dirmap[j][i] = math.atan2(sobelh[j][i], sobelv[j][i])
                if dirmap[j][i] > math.pi / 2:
                    dirmap[j][i] -= math.pi
                if dirmap[j][i] < -math.pi / 2:
                    dirmap[j][i] += math.pi
                dirmapuchar[j][i] = (dirmap[j][i] / math.pi + 0.5) * 255

    return dirmap, sobel
def accumulateinab(edge, width, height, directionmap, minradius, maxradius):

    abspace = np.zeros((width, height))

    for i in range(0, height):
        for j in range(0, width):
            x = minradius * math.cos(directionmap[j][i])
            y = minradius * math.sin(directionmap[j][i])
            dx = 0
            dy = 0
            if((directionmap[j][i] > - (math.pi / 4)) and (directionmap[j][i] < (math.pi / 4))):
                dx = np.sign(x)
                dy = dx * math.tan(directionmap[j][i])
            else:
                dy = np.sign(y)
                dx = dy / math.tan(directionmap[j][i])
            while(math.sqrt(x*x + y*y) < maxradius):
                x1 = int(j + x)
                y1 = int(i - y)
                x2 = int(j - x)
                y2 = int(i + y)
                if x1 in range(0, width) and y1 in range(0, height):
                    abspace[x1][y1] += edge[j][i] / math.sqrt(x*x + y*y)
                if x2 in range(0, width) and y2 in range(0, height):
                    abspace[x2][y2] += edge[j][i] / math.sqrt(x*x + y*y)
                x = x + dx
                y = y + dy
    return abspace

def enhanceabspace(abspace, width, height, threshold):
    mexicanHat = np.array([[0, 0, 0, 0, 0, 0,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0 ],
    [0, 0, 0, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0 ],
    [0, 0,-1,-1,-1,-2,-3,-3,-3,-3,-3,-2,-1,-1,-1, 0, 0 ],
    [0, 0,-1,-1,-2,-3,-3,-3,-3,-3,-3,-3,-2,-1,-1, 0, 0 ],
    [0,-1,-1,-2,-3,-3,-3,-2,-3,-2,-3,-3,-3,-2,-1,-1, 0 ],
    [0,-1,-2,-3,-3,-3, 0, 2, 4, 2, 0,-3,-3,-3,-2,-1, 0 ],
    [-1,-1,-3,-3,-3, 0, 4,10,12,10, 4, 0,-3,-3,-3,-1,-1 ],
    [-1,-1,-3,-3,-2, 2,10,18,21,18,10, 2,-2,-3,-3,-1,-1 ],
    [-1,-1,-3,-3,-3, 4,12,21,24,21,12, 4,-3,-3,-3,-1,-1 ],
    [-1,-1,-3,-3,-2, 2,10,18,21,18,10, 2,-2,-3,-3,-1,-1 ],
    [-1,-1,-3,-3,-3, 0, 4,10,12,10, 4, 0,-3,-3,-3,-1,-1 ],
    [0,-1,-2,-3,-3,-3, 0, 2, 4, 2, 0,-3,-3,-3,-2,-1, 0 ],
    [0,-1,-1,-2,-3,-3,-3,-2,-3,-2,-3,-3,-3,-2,-1,-1, 0 ],
    [0, 0,-1,-1,-2,-3,-3,-3,-3,-3,-3,-3,-2,-1,-1, 0, 0 ],
    [0, 0,-1,-1,-1,-2,-3,-3,-3,-3,-3,-2,-1,-1,-1, 0, 0 ],
    [0, 0, 0, 0,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0 ],
    [0, 0, 0, 0, 0, 0,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0 ]])

    output = cv2.filter2D(abspace, -1, mexicanHat)
    maxval = -1
    for i in range(0, height):
        for j in range(0, width):
            if ((i < 9) or (j < 9) or (i > height - 9) or (j > width - 9)):
                output[j][i] = 0
            if (output[j][i] < 0):
                output[j][i] = 0
            if (output[j][i] > maxval):
                maxval = output[j][i]
        if maxval > 0:
            for k in range(0, height):
                for l in range(0, width):
                    if output[l][k] / maxval * 255 < threshold:
                        output[l][k] = 0

    return output

def accumulateinrspace(abspace, edgemap, dirmap, width, height, image, maxradius, minradius , rthreshold, refinecoordinates):

    contribute = np.zeros((width, height))
    rspace = np.array(maxradius - minradius + 1)
    angle = 0.0
    maxval, x, y = 0.0
    maxpos = 0
    maxa = 0.0
    maxb = 0.0
    maxr = 0.0
    maxv = 0.0

    for i in range(0, height):
        for j in range(0, width):
            if abspace[j][i] > 0:
                maxval = 0
                maxpos = 0

                # start accumulating for radii minradius to maxradius
                for k in range(minradius,maxradius+1, 1):
                    rspace[k-minradius] = 0
                    for l in range(0, k * 2.0 + math.pi, 1):
                        loverkfloat = float(l) / float(k)
                        x = float(k) * math.cos(loverkfloat)
                        y = -float(k) * math.sin(loverkfloat)

                        coord1 = int(i + y + 0.5)
                        coord2 = int(j + x + 0.5)

                        if coord1 in range(0, height) and coord2 in range(0, width):
                            if edgemap[coord2][coord1] > 0:
                                angle = math.atan2(-y, x)
                                if angle < - (math.pi / 2):
                                    angle += math.pi
                                if angle > (math.pi / 2):
                                    angle -= math.pi
                                if(abs(angle - dirmap[coord2][coord1]) < (math.pi / 8)):
                                    rspace[k - minradius] += edgemap[coord2][coord1]
                                    contribute[coord2][coord1] = 255
                    rspace[k - minradius] /= float(k)
                    if rspace[k - minradius] > maxval:
                        maxval = rspace[k - minradius]
                        maxpos = k

                #threshold the r - space
                for k in range(minradius, maxradius+1):
                    rspace[k - minradius] = rspace[k - minradius] / maxval * 255
                    if rspace[k - minradius] + 0.001 < rthreshold:
                        rspace[k - minradius] = 0
                    if j - k >= 0:
                        abspace[j - k][i] = rspace[k - minradius] * 1000
                k = minradius
                maxval = 0
                maxpos = 0
                while k < maxradius:
                    if maxval < rspace[k - minradius]:
                        maxval = rspace[k - minradius]
                        maxpos = k
                    if maxval > 0 and rspace[k - minradius] == 0:
                        if(refinecoordinates):
                            maxv = 0
                            step = 0.5
                            spread = 3.0
                            a = i - spread
                            while a <= float(i) + spread:
                                b = j - spread
                                while b <= float(maxpos) + spread:
                                    r = maxpos - spread
                                    while r <= float(maxpos)+spread:
                                        val = 0
                                        for l in range(0, 1000):
                                            x = float(r) * math.cos(float(l) / 1000.0 * 2 * math.pi)
                                            y = -float(r) * math.sin(float(l) / 1000.0 * 2 * math.pi)
                                            if (y+a) in range(0, height) and (x+b) in range(0, width):
                                                p1 = 0
                                                p2 = 0
                                                p3 = 0
                                                p4 = 0
                                                dirmapX = int(math.floor(x + b))
                                                dirmapY = int(math.floor(y + a))
                                                angle = math.atan2(-y , x)
                                                if(angle < -math.pi / 2):
                                                    angle += math.pi
                                                if angle > math.pi / 2:
                                                    angle -= math.pi
                                                if abs(angle - dirmap[dirmapX][dirmapY]) < math.pi / 8:
                                                    p1 = edgemap[0][0]
                                                if abs(angle - dirmap[dirmapY+1][dirmapX]) < math.pi /8:
                                                    p2 = edgemap[0][0]
                                                if abs(angle - dirmap[dirmapY][dirmapX + 1]) < math.pi / 8:
                                                    p3 = edgemap[0][0]
                                                if abs(angle - dirmap[dirmapY+1][dirmapX+1]) < math.pi / 8:
                                                    p4 = edgemap[0][0]

                                                val += ((1 - math.fmod(y + a, 1)) * p1 + math.fmod(y + a, 1)* p2) * (1 - math.fmod(x + b,1) + (1 - math.fmod(y + a, 1)) * p3 + math.fmod(y + a, 1) * p4)* math.fmod(x+b,1)
                                        if val > maxv:
                                            maxa = a
                                            maxb = b
                                            maxr = r
                                            maxv = val
                                        r += step
                                    b += step
                                a += step
                        else:
                            maxr = maxpos
                            maxa = i
                            maxb = j

                        for l in range(0, maxr * 2.0 * math.pi,1):
                            x = float(maxr) * math.cos(float(l) / float(maxr))
                            y = - float(maxr) * math.sin(float(l) / float(maxr))
                            imgIndeY = int(math.floor(maxa + y + 0.5))
                            imgIndeX = int(math.floor(maxb + x + 0.5))
                            if imgIndeY in range(0, height) and imgIndeX in range(0, width):
                                image[imgIndeY][imgIndeX] = 255 # circle color

                            maxval = 0
                            maxpos = 0
                    k += 1

    return "Done"



def testing():
    red = np.ones((7, 7))
    red = red / 2

    print red

def main2():

    img = cv2.imread("second_derr.png", 0)

    cimg = cv2.imread("second_derr.png")

    circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 10,
                               param1=10, param2=30, minRadius=10, maxRadius=19)
    for i in circles[0, :]:
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imwrite("processed_" + "second_derr.png", cimg)


def main(images):

    mexhatsmall = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])

    averages = np.zeros(images.__len__())
    counter = 0
    for image in images:
        originalImg = cv2.imread(image, 0)

        width, height = originalImg.shape

        for i in range(0, width):
            for j in range(0, height):
                if(originalImg[i][j] >= 92):
                    originalImg[i][j] = 255
                else:
                    originalImg[i][j] = 0
        cv2.imwrite("binary_"+image, originalImg)

        cimg = cv2.cvtColor(originalImg, cv2.COLOR_GRAY2BGR)
        img = smoothing_filter(originalImg, 4)

        cv2.imwrite("smooth_"+image,img)

        sobelv = np.ones((width, height))
        sobelh = np.ones((width, height))

        img, sobelv, sobelh, maxval = sobel_filter(img, 3)

        #dirMap, img = thresholdandfinddirectionmap(img, width, height, sobelh, sobelv, maxval, 200)

        #img = edgeEnhance(originalImg, img, width, height, mexhatsmall)

        #abspace = accumulateinab(img, width, height, dirMap, 5, 15)

        #absspace = enhanceabspace(abspace, width, height, 50)

        cv2.imwrite("sobel_"+image, img)
        img = cv2.imread("sobel_"+image, 0)

        circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 10,
                                   param1=10, param2=30, minRadius=10, maxRadius=20)
        gridSize = 3 #from center 3 pixels in every direction

        for i in circles[0, :]:
            centrY = int(i[0])
            centrX = int(i[1])
            suma = 0

            suma += originalImg[centrX][centrY]
            for j in range(1, gridSize+1):
                suma += originalImg[centrX + j][centrY]
                suma += originalImg[centrX + j][centrY + j]
                suma += originalImg[centrX][centrY + j]
                suma += originalImg[centrX - j][centrY]
                suma += originalImg[centrX - j][centrY - j]
                suma += originalImg[centrX][centrY - 1]
                suma += originalImg[centrX + j][centrY - j]
                suma += originalImg[centrX - j][centrY + j]

            normalizeValue = gridSize * 2 + 1
            normalizeValue *= normalizeValue
            suma = suma / normalizeValue

            #averages[counter] = suma
            #counter = counter + 1

            if(suma <= 50):
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            else:
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 0, 255), 2)
            #cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)


        cv2.imwrite("processed_"+image, cimg)
        print("Processed image: "+image)
    print averages
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#3 7 8 11 12 15
#main(['gajba1.jpg', 'gajba2.jpg', 'gajba3.jpg', 'gajba4.jpg', 'gajba5.jpg', 'gajba6.jpg', 'gajba7.jpg', 'gajba8.jpg',
#     'gajba9.jpg', 'gajba10.jpg', 'gajba11.jpg', 'gajba12.jpg', 'gajba13.jpg', 'gajba14.jpg', 'gajba15.jpg', 'gajba16.jpg', 'gajba17.jpg'])
#main2()

#main(['gajba3.jpg', 'gajba7.jpg', 'gajba8.jpg', 'gajba11.jpg', 'gajba12.jpg', 'gajba15.jpg'])
#main(['gajba3.jpg'])

def main2():
    infrastructure = np.array([2, 3, 2])
    initial_values = np.array([1, 0])
    network = neural.Network()

    nodeCount = infrastructure.sum()
    for i in range(0, infrastructure.__len__()):

        for layerCount in range(0, infrastructure[i]):
            node = neural.Node()
            if i == 0:
                node.value = initial_values[i]
            node.bias = randint(-10, 10)
            network.add_node("layer" + str(i + 1) + "node"+str(layerCount), node)



    network.show_network()

def main3():
    cap = cv2.VideoCapture('videos4/VIDEO0017.mp4')
    print cv2.__version__
    frameCounter = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        trainingData = []
        flag, frame = cap.read()
        if flag == 0:
            break
        #cv2.imshow("Video", frame)
        key_pressed = cv2.waitKey(10)  # Escape to exit
        if key_pressed == 27:
            break

        originalImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #originalImg = cv2.imread(frame, 0)

        width, height = originalImg.shape
        cimg = cv2.cvtColor(originalImg, cv2.COLOR_GRAY2BGR)

        img = smoothing_filter(originalImg, 4)

        #cv2.imwrite("smooth_" + image, img)

        sobelv = np.ones((width, height))
        sobelh = np.ones((width, height))

        img, sobelv, sobelh, maxval = sobel_filter(img, 3)

        # dirMap, img = thresholdandfinddirectionmap(img, width, height, sobelh, sobelv, maxval, 200)

        # img = edgeEnhance(originalImg, img, width, height, mexhatsmall)

        # abspace = accumulateinab(img, width, height, dirMap, 5, 15)

        # absspace = enhanceabspace(abspace, width, height, 50)

        cv2.imwrite("sobel.png", img)
        img = cv2.imread("sobel.png", 0)

        circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 10,
                                   param1=10, param2=30, minRadius=10, maxRadius=20)
        gridSize = 3  # from center 3 pixels in every direction


        if circles is None:
            cv2.imshow("Video", cimg)
            #cv2.imwrite("frames/frame" + str(frameCounter) + ".png", cimg)
            frameCounter += 1
            continue
        size, la = circles[0].shape

        nearSomeone = 0
        circlesNotValidIndexes = []
        for i in range(0, size):
            currentX = circles[0][i][0]
            currentY = circles[0][i][1]
            nearSomeone = 0
            j = 0
            while j < size and nearSomeone < 2:
                if(i == j):
                    j += 1
                    continue
                elseX = circles[0][j][0]
                elseY = circles[0][j][1]
                if math.fabs(currentX - elseX) <= 30:
                    nearSomeone = nearSomeone + 1
                if math.fabs(currentY - elseY) <= 30:
                    nearSomeone = nearSomeone + 1
                j += 1
            if nearSomeone < 2:
                circles[0][i] = (0, 0, 0)
                #circlesNotValidIndexes.append(i)

        circles_x_coords = []
        circles_y_coords = []
        circles_radius = []
        circles_flags = []

        centers = []

        counter = 0
        for i in circles[0, :]:
            if counter in circlesNotValidIndexes:
                counter += 1
                continue

            centrY = int(i[0])
            circles_x_coords.append(centrY)
            centrX = int(i[1])
            circles_y_coords.append(centrX)
            radius = int(i[2])
            circles_radius.append(radius)

            if(centrY == 0 and centrX == 0):
                counter += 1
                continue
            trData = [centrY, centrX, radius, 0, 0]

            suma = 0
            normalizeValue = gridSize * 2 + 1
            suma += originalImg[centrX][centrY]
            for j in range(1, gridSize + 1):
                if centrX + j < width and centrX - j >= 0 and centrY + j < width and centrY-j >= 0:
                    suma += originalImg[centrX + j][centrY]
                    normalizeValue = normalizeValue - 8

                    suma += originalImg[centrX + j][centrY + j]
                    suma += originalImg[centrX][centrY + j]
                    suma += originalImg[centrX - j][centrY]
                    suma += originalImg[centrX - j][centrY - j]
                    suma += originalImg[centrX][centrY - 1]
                    suma += originalImg[centrX + j][centrY - j]
                    suma += originalImg[centrX - j][centrY + j]
#
            if suma == 0:
                continue

            normalizeValue *= normalizeValue
            suma = suma / normalizeValue
#
            # averages[counter] = suma
            # counter = counter + 1

            trData[4] = suma
            if (suma <= 13):
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
                trData[3] = 1
                center = (i[0], i[1], 1)
                circles_flags.append(1)
            else:
                cv2.circle(cimg, (i[0], i[1]), i[2], (0, 0, 255), 2)
                # cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
                trData[3] = 0
                center = (i[0], i[1], 0)
                circles_flags.append(0)
            centers.append(center)

            trainingData.append(trData)
            counter += 1
        dtype = [('x', float), ('y', float), ('good', int)]

        npArray = np.array(centers, dtype)

        npArray = np.sort(npArray, order='y')
        npArray = np.sort(npArray, order='x')


        indexes = ''
        redFlagCounter = 0
        for i in range(0, npArray.size):
            sortX,sortY,flag = npArray[i]
            if(flag == 0):
                redFlagCounter += 1
                indexes += str(i+1) + ','
            if (i == npArray.size - 1):
                indexes = indexes[:-1]

        cv2.putText(cimg, 'Found:' + str(redFlagCounter), (0, 30), font, 1, (0, 0, 0), 2)
        cv2.putText(cimg, 'Indexes:' + indexes, (0, 60), font, 1, (0, 0, 0), 2)
        cv2.imshow("Video", cimg)

        #cv2.imwrite("frames/frame"+str(frameCounter)+".png",cimg)
        frameCounter += 1
        numpyArrayTransform = np.array(trainingData)

        time.sleep(1)
        del trainingData[:]
        # Write the array to disk
        #with file('test.txt', 'a') as outfile:
            #outfile.write("#Sample number:("+str(frameCounter)+")\n")
            #np.savetxt(outfile, numpyArrayTransform, fmt='%-7.2f')

    cap.release()
    cv2.destroyAllWindows()

def plotVideoSamples():
    with file('test.txt', 'r') as outfile:

        result = np.loadtxt(outfile)
        print result
def testing():
    testSample = [628.00, 609.00, 372.00, 609.00, 260.00]
    mean = 0
    for i in range(0, testSample.__len__()):
        for j in range(0, testSample.__len__()):
            if(i == j):
                continue

    #print diffResult

#main(['bwi.jpg'])
#main3()
#main2()
#plotVideoSamples()
#testing()

def pixelsInCircle(xCentar, yCentar, radius, picture): # mora da se prosledi gray_scale image
    # ove 2 promenjive su za rucno izracunavanje, svuda odkomentarisati ako treba to
    sum = 0
    numberOfPixels = 0
    image = copy.deepcopy(picture)
    array = []
    width = np.size(picture, 0)
    height = np.size(picture, 1)
    for x in range(xCentar - radius, xCentar+1):
        for y in range(yCentar - radius, yCentar+1):
            if ((x - xCentar)*(x - xCentar) + (y - yCentar)*(y - yCentar) <= radius*radius):
                xSym = xCentar - (x - xCentar)
                ySym = yCentar - (y - yCentar)
                #(x, y), (x, ySym), (xSym , y), (xSym, ySym) are in the circle
                # Ovo je za slucaj bez uslova, ako zatreba brzina izvrsavanja
                # -----------------------------------------------------
                sum += int(picture[x, y]) * int(picture[x, y])
                sum += int(picture[x, ySym]) * int(picture[x, ySym])
                sum += int(picture[xSym, y]) * int(picture[xSym, y])
                sum += int(picture[xSym, ySym]) * int(picture[xSym, ySym])
                numberOfPixels += 4
                # -----------------------------------------------------
                if ((x >= 0) & (width > x) & (y >= 0) & (height > y)):
                    image[x, y] = 255
                    array.append(int(picture[x, y]))
                    sum += int(picture[x, y]) * int(picture[x, y])
                    numberOfPixels += 1
                if ((x >= 0) & (width > x) & (ySym >= 0) & (height > ySym)):
                    image[x, ySym] = 255
                    array.append(int(picture[x, ySym]))
                    sum += int(picture[x, ySym]) * int(picture[x, ySym])
                    numberOfPixels += 1
                if ((xSym >= 0) & (width > xSym) & (y >= 0) & (height > y)):
                    image[xSym, y] = 255
                    array.append(int(picture[xSym, y]))
                    sum += int(picture[xSym, y]) * int(picture[xSym, y])
                    numberOfPixels += 1
                if ((xSym >= 0) & (width > xSym) & (ySym >= 0) & (height > ySym)):
                    image[xSym, ySym] = 255
                    array.append(int(picture[xSym, ySym]))
                    sum += int(picture[xSym, ySym]) * int(picture[xSym, ySym])
                    numberOfPixels += 1
    #cv2.imwrite('krugTest.png', image) # sacuva sliku sa iscrtanim belim krugom, za proveru radiusa

    # cuva u fajl vrednosti 0-255 za izmenjenu sliku i original, za proveru vrednosti
    # povecati for petlje za duzinu slike ako zatreba, duze traje upisivanje u txt !!!
    #for i in range(0, width/3):
    #    for j in range(0, height/3):
    #        f = open('image.txt', 'a')
    #        f.write(str(image[i,j]) + " ")
    #        f.close()
    #        f = open('picture.txt', 'a')
    #        f.write(str(picture[i,j]) + " ")
    #        f.close()
    #    f = open('image.txt', 'a')
    #    f.write('\n')
    #    f.close()
    #    f = open('picture.txt', 'a')
    #    f.write('\n')
    #    f.close()

    #pom2 = int(np.mean(array, dtype=np.float64))
    pom = int(np.sqrt(sum/numberOfPixels)) # ako treba da se vrati rucno izracunavanje
    return pom


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setup_camera()
        self.StartButton.clicked.connect(self.video_processing)
        #self.StartButton.clicked.connect(self.open)
        self.SetStateGoodButton.clicked.connect(self.array_mean_values) # racuna srednje vrednosti za nizove cep i nonCep

        self.videoFullyProcessed = False
        self.nextFrameRequested = True
        self.overrideFlag = False
        self.SetStateBadButton.clicked.connect(self.override)

        self.VideoView.mousePressEvent = self.drawCircle
        self.VideoView.mouseMoveEvent = self.showCoords
        self.VideoView.wheelEvent = self.next_frame
        self.scene = QtGui.QGraphicsScene()
        self.VideoView.setScene(self.scene)
        #self.capture = None
        self.currentImage = None # cuva trenutni ucitan frame sa UI-a
        self.cepValues = [] # vrednosti srednjih vrednosti piksela za flase sa cepovima
        self.nonCepValues = []  # vrednosti srednjih vrednosti piksela za flase sa bez cepova
    
    def array_mean_values(self): # racuna srednje vrednosti za cep i nonCep, i upisuje u 2 txt-a (cep i nonCep) sve vrednosti niza i njihove srednje vrednosti
        val1 = int(np.mean(self.cepValues))
        val2 = int(np.mean(self.nonCepValues))

        print "\nCep mean: " + str(val1)
        print "nonCep mean: " + str(val2)

        for i in range(0, len(self.cepValues)):
            f = open('cepValues.txt', 'a')
            f.write(str(self.cepValues[i]) + "\n")
            f.close()
        f = open('cepValues.txt', 'a')
        f.write("\n Neam: " + str(val1) + " ")
        f.close()
        for i in range(0, len(self.nonCepValues)):
            f = open('nonCepValues.txt', 'a')
            f.write(str(self.nonCepValues[i]) + "\n")
            f.close()
        f = open('nonCepValues.txt', 'a')
        f.write("\n Neam: " + str(val2) + " ")
        f.close()

    def override(self):
        self.overrideFlag=not self.overrideFlag
        self.nextFrameRequested = True
    def video_processing(self):
        fileName, _ = QtGui.QFileDialog.getOpenFileName(self, "Open File",
                                                        QtCore.QDir.currentPath())
        if fileName:
            self.capture = cv2.VideoCapture(fileName)
            
            if self.capture is None:
                QtGui.QMessageBox.information(self, "Image Viewer",
                                          "Cannot load %s." % fileName)
                return
        self.timer.start(30)

    def next_frame(self, event):
        self.nextFrameRequested = True

    def showCoords(self, point):
        xPoint = point.x()
        yPoint = point.y()
        self.coordLabel.setText("X: "+ str(xPoint) + "  Y: " + str(yPoint))

    def drawCircle(self, point):

        pressedButtons = point.buttons()
        pen = QtGui.QPen()
        if pressedButtons == QtCore.Qt.LeftButton:
            pen.setColor("Green")
        elif pressedButtons == QtCore.Qt.RightButton:
            pen.setColor("Red")
        else:
            pen.setColor("White")

        pen.setWidth(3)
        brush = QtGui.QBrush()
        #brush.setColor("Red")  cv2.cv.CV_BGR2RGB  cv2.COLOR_BGR2GRAY

        xPoint = point.x()
        yPoint = point.y()

        #flag, frame = self.capture.read()
        #frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
        #frame = cv2.flip(frame, 1)
        frame = cv2.flip(self.currentImage, 1)
        if pressedButtons == QtCore.Qt.LeftButton:
                #gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #pixelsInCircle(point.x(), point.y(), 50, gray_image) 
                val = pixelsInCircle(point.y(), point.x(), 15, gray_image)
                self.nonCepValues.append(val)
                print "nonCep: " + str(val)
        if pressedButtons == QtCore.Qt.RightButton:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                val = pixelsInCircle(point.y(), point.x(), 12, gray_image)
                self.cepValues.append(val)
                print "Cep: " + str(val)

        self.scene.addEllipse(xPoint-10, yPoint-10, 20, 20, pen, brush)

        self.VideoView.setScene(self.scene)

    def open(self):
        fileName, _ = QtGui.QFileDialog.getOpenFileName(self, "Open File",
                                                        QtCore.QDir.currentPath())
        if fileName:
            image = QtGui.QImage(fileName)
            if image.isNull():
                QtGui.QMessageBox.information(self, "Image Viewer",
                                              "Cannot load %s." % fileName)
                return
            self.scene.setSceneRect(image.rect())
            self.scene.addPixmap(QtGui.QPixmap.fromImage(image))
            self.VideoView.setScene(self.scene)

            self.scaleFactor = 1.0
            self.VideoView.autoFillBackground()

            self.VideoView.fitInView(self.VideoView.rect())

    def setup_camera(self):
        """Initialize camera.
        """
        #self.capture = cv2.VideoCapture('videos4/VIDEO0017.mp4')
        
        #self.capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, self.video_size.width())
        #self.capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.display_video_stream)

    def display_video_stream(self):
        if self.nextFrameRequested:

            flag, frame = self.capture.read()
            self.currentImage = frame
            if flag == 0:
                self.videoFullyProcessed = True
                self.timer.stop()
                self.capture.release()

                return

            #test-----------------------
            #pressedButtons = point.buttons()
            #if pressedButtons == QtCore.Qt.LeftButton:
                #gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #pixelsInCircle(point.x(), point.y(), 30, gray_image)

            self.videoFullyProcessed = False
            frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
            frame = cv2.flip(frame, 1)
            image = QtGui.QImage(frame, frame.shape[1], frame.shape[0],
                           frame.strides[0], QtGui.QImage.Format_RGB888)

            self.scene.clear()
            self.scene.update()
            self.scene.setSceneRect(image.rect())
            self.scene.addPixmap(QtGui.QPixmap.fromImage(image))
            self.VideoView.setScene(self.scene)

            self.scaleFactor = 1.0
            self.VideoView.autoFillBackground()

            self.VideoView.fitInView(self.VideoView.rect())

            if(self.overrideFlag):
                return
            self.nextFrameRequested = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    frame = MainWindow()
    frame.show()
    app.exec_()