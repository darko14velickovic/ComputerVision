import numpy as np
from random import randint
import cv2
import time
#from PIL import Image
def checkPixelCoords(x,y,imgX,imgY):
    return x >= 0 and y >= 0 and x < imgX and y < imgY
def main(images):
    for image in images:
        img = cv2.imread(image, 1)
        # blur = cv2.GaussianBlur(img, (0, 0), 2)
        img = cv2.medianBlur(img, 5)
        edges = cv2.Canny(img, 100, 200)
        (x,y) = img.shape
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #cv2.imshow(image, edges)
        circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 10,
                                   param1=10, param2=30, minRadius=10, maxRadius=20)

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle

            #crosshair of pixels
            sumXup = 0
            sumXdown = 0
            sumYup = 0
            sumYdown = 0

            xupCount = 0
            xdownCount = 0
            yupCount = 0
            ydownCount = 0

            for j in range(0, 2, 1):
                if checkPixelCoords(i[0]+j, i[1], x, y):
                    sumXup += img[i[0] + j][i[1]]
                    xupCount += 1
                if checkPixelCoords(i[0]-j, i[1], x, y):
                    sumXdown += img[i[0]-j][i[1]]
                    xdownCount += 1
                if checkPixelCoords(i[0], i[1]+j, x, y):
                    sumYup += img[i[0]][i[1]+j]
                    yupCount += 1
                if checkPixelCoords(i[0], i[1]-j, x, y):
                    sumYdown += img[i[0]][i[1]-j]
                    ydownCount += 1

            if xupCount == 0:
                xupCount = 1
            if xdownCount == 0:
                xdownCount = 1
            if yupCount == 0:
                yupCount = 1
            if ydownCount == 0:
               ydownCount = 1

            medianXup = sumXup / xupCount
            medianXdown = sumXdown / xdownCount
            medianYup = sumYup / yupCount
            medianYdown = sumYdown / ydownCount

            result = (medianXup + medianXdown + medianYup + medianXdown) / 4
            color = (0, 255, 0)
            if result < 195:
                cv2.circle(cimg, (i[0], i[1]), 2, color, 3)
                cv2.circle(cimg, (i[0], i[1]), i[2], color, 2)
        cv2.imshow(image, cimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
main(['gajba17.jpg'])
#'gajba1.jpg','gajba2.jpg','gajba3.jpg','gajba4.jpg','gajba5.jpg','gajba6.jpg',
#      'gajba7.jpg','gajba8.jpg','gajba9.jpg','gajba10.jpg','gajba11.jpg','gajba12.jpg','gajba13.jpg','gajba14.jpg','gajba15.jpg','gajba16.jpg',
