import cv2
import numpy as np


def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('lowH','image',0,255,nothing)

cv2.createTrackbar('lowV','image',255,255,nothing)
cv2.createTrackbar('lowS','image',0,255,nothing)

cv2.createTrackbar('highH','image',0,255,nothing)
cv2.createTrackbar('highS','image',255,255,nothing)
cv2.createTrackbar('highV','image',255,255,nothing)


img=cv2.imread('t.JPG')
img=cv2.resize(img,(600,600))
# mask = np.zeros_like(img)

while True:
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask2 = cv2.inRange(img, lower_hsv, higher_hsv)
    frame = cv2.bitwise_and(img, img, mask=mask2)

    print(mask2)
    # print(ihighH)

    cv2.imshow('window',img)
    cv2.imshow('mask',frame)
    # cv2.imshow('gray',gray)
    # cv2.imshow('canny',canny)
    # cv2.imshow('blur',blur)
    # cv2.imshow('thres',thres)
    cv2.waitKey(1)

