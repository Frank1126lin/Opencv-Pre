#! /usr/bin/env python3
# *_* coding: utf-8 *_*
# @File  : LableOCR.py
# @Author: Frank1126lin
# @Date  : 5/6/2020

import numpy as np
import cv2
import pytesseract as tess

drawing = False
ix, iy = -1, -1
ptx, pty = 0, 0
def Drawit(event, x, y, flags, param):
    global ix, iy, drawing, ptx, pty
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    # elif event == cv2.EVENT_MOUSEMOVE:
    #     if drawing is True:
    #         cv2.rectangle(img,(ix, iy), (x,y), (0,255,0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ptx = x
        pty = y
        cv2.rectangle(img, (ix, iy), (ptx, pty), (0, 255, 0), 2)

def transgray(img):
    # print(img.shape)
    (h, w) = img.shape
    dst = np.zeros((h, w, 1), np.uint8)
    for i in range(1,h):
        for j in range(0,w):
            grayPixel = img[i, j]
            dst[i,j] = 255-grayPixel
    return dst

def cutimage2string(img, ix, iy, ptx, pty):
    '''
    img :opencv 读取的图像
    ix,iy,ptx, pty 图像的截图区域
    return: 文字识别的string
    '''
    img2 = img.copy()
    imgcut = img2[iy:pty,ix:ptx]
    imgcutgray = cv2.cvtColor(imgcut, cv2.COLOR_BGR2GRAY)
    cv2.imshow('imgcutgray', imgcutgray)
    dst = transgray(imgcutgray)
    # dst = imgcutgray
    rst, binary = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow('imgbin', binary)
    string = tess.image_to_string(binary)
    if string is None:
        return None
    return string

img = cv2.imread('./LOGO.bmp')
print(img.shape)
(h, w, d) = img.shape
# print(h,w)
img = cv2.resize(img,(640,int(w/(h/640))))
cv2.namedWindow('image')
cv2.setMouseCallback('image',Drawit)


while True:
    cv2.imshow('image',img)
    # print(ix, iy, ptx, pty)

    if ptx != 0:
        # print(ptx)
        cv2.imshow('cutimg', img[iy:pty, ix:ptx])
        string = cutimage2string(img, ix, iy, ptx, pty)
        string_list = list(string)
        if string:
            print(string)
            for i in range(len(string_list)):
                cv2.putText(img, string_list[i], (ix+i*15, iy - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
        else:
            print("[INFO] No text!")

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
cv2.destroyAllWindows()









