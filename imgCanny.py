#! /usr/bin/env python3
# *_* coding: utf-8 *_*
# @File  : imgCanny.py
# @Author: Frank1126lin
# @Date  : 4/7/20

import cv2
import numpy as np


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(imgCnt, cnt, -1, (0,255,0),1)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)


path = "./img/3.bmp"
img = cv2.imread(path)
imgCnt = img.copy()
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.resize(img,(640, 480))
# cv2.resize(imgGray, (640, 480))
# imgBlur = cv2.GaussianBlur(img, (3,3),1)
imgCanny = cv2.Canny(img, 50, 50)
getContours(imgCanny)

# imgBlank = np.zeros_like(img)






cv2.imshow("img", img)
cv2.imshow("imgCanny", imgCanny)
cv2.imshow("imgCnt", imgCnt)
cv2.waitKey(0)