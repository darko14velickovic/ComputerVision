import numpy as np
# from random import randint
import cv2
import math

# import time
# from PIL import Image


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

    img = cv2.imread("second_derr.png",0)

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
        cimg = cv2.cvtColor(originalImg, cv2.COLOR_GRAY2BGR)
        img = smoothing_filter(originalImg, 4)

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

main(['gajba3.jpg', 'gajba7.jpg', 'gajba8.jpg', 'gajba11.jpg', 'gajba12.jpg', 'gajba15.jpg'])