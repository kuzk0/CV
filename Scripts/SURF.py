import numpy as np
import cv2 as cv

img = cv.imread('test.jpg', cv.IMREAD_GRAYSCALE)
# Создание объекта SURF. Вы можете указать параметры здесь или позже.
# Здесь я установил пороговое значение гессиана равным 400
surf = cv.xfeatures2d.SURF_create(400)
# Находим ключевые точки и дескрипторы напрямую
kp, des = surf.detectAndCompute(img,None)
print( len(kp) )
print( surf.getHessianThreshold() )
surf.setHessianThreshold(50000)
kp, des = surf.detectAndCompute(img,None)
print( len(kp) )

img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
cv.imshow(img2)
cv.waitKey(0)
cv.destroyAllWindows()
