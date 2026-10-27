import numpy as np
import cv2 as cv

img = cv.imread('test.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray, None)

img=cv.drawKeypoints(gray,kp,img)

cv.imshow('SIFT Keypoints',img) 
cv.waitKey(0)
cv.destroyAllWindows()