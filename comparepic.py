#! /usr/bin/env python3
# *_* coding: utf-8 *_*
# @File  : comparepic.py
# @Author: Frank1126lin
# @Date  : 2020/4/2


# 导入相关包
from skimage.metrics import structural_similarity
import imutils
import numpy as np
import cv2

# 读取文件并转换为灰度
imageA = cv2.imread("./img/3.bmp")
imageB = cv2.imread("./img/4.bmp")

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# 计算灰度之间的相似度系数
score, diff = structural_similarity(grayA, grayB, full=True)
diff = (diff*255).astype("uint8")
print("SSIM:{}".format(score))

if score == 1.0:
    print("图片相同！")
else:
    # 找到不同点的轮廓
    # thresh = cv2.threshold(diff,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(diff,150,255,cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    kernel = np.ones((5,5), np.uint8)
    dst = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)


    # 找到区域，放置矩形
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # cv2.rectangle(imageA, (x,y), (x+w, y+h), (255,0,0),2)
        cv2.rectangle(imageB, (x,y), (x+w, y+h), (255,0,0),2)

    # 展现，保存
    cv2.namedWindow("Origin", 0)
    cv2.resizeWindow("Origin", 640, 480)
    cv2.imshow("Origin",imageA)

    cv2.namedWindow("Test", 0)
    cv2.resizeWindow("Test", 640, 480)
    cv2.imshow("Test",imageB)

    cv2.namedWindow("Diff", 0)
    cv2.resizeWindow("Diff", 640, 480)
    cv2.imshow("Diff",diff)

    cv2.namedWindow("Dst", 0)
    cv2.resizeWindow("Dst", 640, 480)
    cv2.imshow("Dst",dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()