import cv2
import numpy as np

n = 100

sift = cv2.SIFT_create()


img1rgb = cv2.imread('Scripts/matching/1.jpg')
img2rgb = cv2.imread('Scripts/matching/2.jpg')

img1 = cv2.cvtColor(img1rgb, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2rgb, cv2.COLOR_BGR2GRAY)


#--------------------------------------------------
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)

# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
# good = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good.append([m])
        
# img3 = cv2.drawMatchesKnn(
#     img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
# )
#-------------------------------------------------





#---------------------------------------------------------
# orb = cv2.ORB_create()

# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)

# img3 = cv2.drawMatches(
#     img1,
#     kp1,
#     img2,
#     kp2,
#     matches[:n],
#     None,
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
# )
#-------------------------------------------------------


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

FLANN_INDEX_LSH = 6
index_params = dict(
    algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2
)

search_params = dict(checks=50)

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

matchesMask = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matchesMask,
    flags=cv2.DrawMatchesFlags_DEFAULT,
)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)




cv2.imshow('test', img3)

cv2.waitKey(0)
cv2.destroyAllWindows()