import numpy as np
from random import randint
import cv2
import time
#from PIL import Image

def main(images):
    for image in images:
        img = cv2.imread(image, 0)
        # blur = cv2.GaussianBlur(img, (0, 0), 2)
        img = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 10,
                                   param1=10, param2=30, minRadius=10, maxRadius=20)

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow(image, cimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
main(['gajba1.jpg','gajba2.jpg','gajba3.jpg','gajba4.jpg','gajba5.jpg','gajba6.jpg',
      'gajba7.jpg','gajba8.jpg','gajba9.jpg','gajba10.jpg','gajba11.jpg','gajba12.jpg','gajba13.jpg','gajba14.jpg','gajba15.jpg','gajba16.jpg','gajba17.jpg'])
