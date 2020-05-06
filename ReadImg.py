#! /usr/bin/env python3
# *_* coding: utf-8 *_*
# @File  : ReadImg.py
# @Author: Frank1126lin
# @Date  : 2020/5/6

import cv2
import numpy as np
import PySimpleGUI as sg

sg.theme('Dark Blue 3')

layout = [
    [sg.Image(data='', size=(640,480),key='image')],
    [sg.Text("请选择文件")],
    [sg.Input(key="input", size=(80,1))],
    [sg.FileBrowse(target="input"), sg.OK()],
]

window = sg.Window('图片浏览', layout=layout, location=(640, 480), finalize=True)

while True:
    event, values = window.read()
    # print(event, values)
    if event in (None, "Quit"):
        break
    img = cv2.imread(values["input"])
    # print(img.shape)
    (w, h, l) = img.shape
    img = cv2.resize(img, (640, int(h/(w/640))))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = cv2.imencode('.png', img)
    # print(t)
    imgbytes = t[1].tobytes()
    # print(imgbytes)
    window['image'].update(data=imgbytes)

window.close(); del window

