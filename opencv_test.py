#! /usr/bin/env python3
# *_* coding: utf-8 *_*
# @File  : opencv_test2.py
# @Author: Frank1126lin
# @Date  : 4/1/20

import cv2
import numpy as np


# def model_match(search_image, tmp_image, threshold):
#     img = cv2.imread(search_image)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将搜索图片转化为灰度图
#
#     tmp_img = cv2.imread(tmp_image)#读取模板图片的灰度图
#     tmp_img_gray = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
#     h, w = tmp_img_gray.shape
#
#     res = cv2.matchTemplate(img_gray, tmp_img_gray, cv2.TM_CCOEFF_NORMED)#采用标准相关匹配 方法度量相似度
#     print(res.shape)
#     loc = np.where(res>threshold)#返回每一个维度的坐标值， 比如返回值为[1, 2, 3],[1, 2, 4] 表明（1， 1）（2， 2）（3， 4）这三个坐标点的相似度大于阈值
#     print(loc)
#     for pt in zip(*loc):#zip(*loc)反解析为坐标值
#         print(pt)
#         cv2.rectangle(img, pt[::-1], (pt[1] + w, pt[0] + h), (0,255,0), 2) #在搜索图片上绘制矩形框
#     cv2.imshow('Detected',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# if __name__ == "__main__":
#     img = './img/3.bmp'
#     tmp_image = './img/tmp3.png'
#     threshold = 0.8
#     model_match(img, tmp_image, threshold)


img =  cv2.imread("./img/10.png")
img = cv2.resize(img, (640,480))

# imgCrop = img[100:400, 110:460]
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgCanny = cv2.Canny(imgGray,40,40)
# imgMor = cv2.morphologyEx(imgCanny, cv2.MORPH_OPEN, (5,5))
# cv2.imshow("img", img)
cv2.imshow("imgGray", imgGray)
cv2.imshow("imgCanny", imgCanny)
# cv2.imshow("imgMor", imgMor)
cv2.imwrite("cap.jpg", imgCanny)
cv2.waitKey(0)