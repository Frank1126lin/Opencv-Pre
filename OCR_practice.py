#! /usr/bin/env python3
# *_* coding: utf-8 *_*
# @File  : OCR_practice.py
# @Author: Frank1126lin
# @Date  : 4/11/20

'''
通过找到每个字符的外接轮廓，然后再截取外接轮廓，传递给tesseract进行识别，最后实现打印在字符上
'''
import re
import cv2
import numpy as np
import pytesseract as tess


# 反转图像，备用
def transgray(img):
    # print(img.shape)
    (h, w) = img.shape
    dst = np.zeros((h, w, 1), np.uint8)
    for i in range(1,h):
        for j in range(0,w):
            grayPixel = img[i, j]
            dst[i,j] = 255-grayPixel
    return dst

# 找到图片里的所有外界轮廓
path = "./label/gpu.jpg"
# 图像读取
img = cv2.imread(path)
# 图像灰度处理+二值化处理
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgGray = transgray(imgGray)
ret, binary = cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 找到图像轮廓
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 绘制蓝色轮廓
cv2.drawContours(img, contours, -1, (255,0,0),2)

# 找到所有位置信息
for i in range(0,len(contours)):
    # area = cv2.contourArea(contours[i])  # 计算轮廓面积
    # # if area > 1000:
    # #     pass
    # # elif area < 200:
    # #     pass
    # # else:
    x, y, w, h = cv2.boundingRect(contours[i])  # 找到所有点位
    # 切割检测出的矩形区域的二值化图片
    imgcut = binary[y:y+h+2, x:x+w+2]
    # cv2.imshow("imgcut", imgcut)
    # 将切割后的图片进行tess识别
    string = tess.image_to_string(imgcut)
    matchobj = re.match('[a-zA-Z0-9]',string)
    if matchobj:
        # 成功则画绿框并把字符写到轮廓上
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  # 绘制外接矩形绿色轮廓
        cv2.putText(img, string, (x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)

    else:
        # 失败画红框
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)  # 绘制外接矩形红色轮廓


cv2.namedWindow("img", 0)
cv2.resizeWindow("img",640,480)
cv2.imshow("img", img)

cv2.namedWindow("imgThresh", 0)
cv2.resizeWindow("imgThresh",640,480)
cv2.imshow("imgThresh", binary)

cv2.waitKey(0)
cv2.destroyAllWindows()




