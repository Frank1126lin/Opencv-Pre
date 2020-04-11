#! /usr/bin/env python3
# *_* coding: utf-8 *_*
# @File  : DocScanner.py
# @Author: Frank1126lin
# @Date  : 4/9/20

import cv2
import numpy as np

imgWidth = 640
imgHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, imgWidth)
cap.set(4, imgHeight)
cap.set(10, 150)


def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny =cv2.Canny(imgBlur,50,50)
    # kernel = np.ones((5,5))
    # imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    # imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgCanny
    # return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            # cv2.drawContours(imgCnt, cnt, -1, (0,255,0),1)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (0,255,0),20)
    return biggest


def reorder(points):
    points = points.reshape((4,2))
    # print("points",points)
    sum = points.sum(1)
    # print("sum", sum)
    pointsNew = np.zeros((4,1,2), np.int32)
    pointsNew[0] = points[np.argmin(sum)]
    pointsNew[3] = points[np.argmax(sum)]
    diff = np.diff(points, axis=1)
    # print("diff", diff)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]
    # print("pointsNew",pointsNew)
    return pointsNew

def getWrap(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0], [imgWidth,0], [0, imgHeight], [imgWidth,imgHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))

    return imgOutput

while True:
    success, img = cap.read()
    img = cv2.resize(img, (imgWidth,imgHeight))
    # img = cv2.imread("./11.jpg")
    imgContour = img.copy()

    imgThres = preProcessing(img)
    biggest = getContours(imgThres)
    # print(biggest)
    if biggest.size != 0:
        imgWrap = getWrap(img, biggest)
        print("Working>>>")
        cv2.imshow("Result", imgWrap)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        print(">")
        print(">>")
        print(">>>")


