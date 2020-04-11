#! /usr/bin/env python3
# *_* coding: utf-8 *_*
# @File  : opencv_multitmp.py
# @Author: Frank1126lin
# @Date  : 4/5/20


import cv2
import numpy as np
import imutils


# 读取模板图片
template = cv2.imread("./img/tmp1.png")
# 转换为灰度图片
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
cv2.imshow("Template", template)
# 执行边缘检测
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]

# 显示模板
cv2.imshow("tmpCanny", template)

# 读取测试图片并将其转化为灰度图片
img = cv2.imread("./img/3.bmp")
# image = cv2.resize(image, (1280, 900))
# img = image[170:720, 180:1000]
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
FOUND = None

# 循环遍历不同的尺度
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    # 根据尺度大小对输入图片进行裁剪
    resized = imutils.resize(img, width = int(img.shape[1] * scale))
    r = img.shape[1] / float(resized.shape[1])

    # 如果裁剪之后的图片小于模板的大小直接退出
    if resized.shape[0] < tH or resized.shape[1] < tW:
        break

    # 首先进行边缘检测，然后执行模板检测，接着获取最小外接矩形
    edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    # 结果可视化
    # 绘制矩形框并显示结果
    # clone = np.dstack([edged, edged, edged])
    # cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (0, 255, 0), 2)
    # cv2.imshow("Edge", clone)
    # cv2.waitKey(0)
    #如果发现一个新的关联值则进行更新
    if FOUND is None or maxVal > FOUND[0]:
        FOUND = (maxVal, maxLoc, r)

# 计算测试图片中模板所在的具体位置，即左上角和右下角的坐标值，并乘上对应的裁剪因子
(_, maxLoc, r) = FOUND
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

# 绘制并显示结果
cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()