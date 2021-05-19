#! /usr/bin/env python3
# *_* coding: utf-8 *_*
# @File  : image_cut_wrap_highpass.py
# @Author: FRANK1126LIN
# @Date  : 5/18/21
'''
给定输入的图片地址，返回图像切割-视角变换-高反差处理，每个功能一个模块
'''


import cv2
import numpy as np
import os

def pre_process(img):
    '''

    :param img: 输入cv2读取的图像
    :return: 预处理后的图像内容
    '''

    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2BGRA)  # 灰度处理
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)  # 锐化
    img_canny = cv2.Canny(img_blur, 5, 200)  # 边缘线
    kernel = np.ones((1, 1))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)  # 图片膨胀2次,保留连续边缘线
    img_thres = cv2.erode(img_dilate, kernel, iterations=1)  # 图片腐蚀一次，删除多余细小边缘
    return img_thres


def get_coutours(img):
    points = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找轮廓
    for i, cont in enumerate(contours):
        area = cv2.contourArea(cont)  # 计算每个轮廓的大小
        #print("轮廓大小：{}",area)  # 打印轮廓大小
        if area > 8000:  # 如果轮廓大于8000像素平方
            # print("轮廓大小：{}", area)  # 打印轮廓大小
            peri = cv2.arcLength(cont, True)  # 计算轮廓长度
            # print("peri:", peri)
            approx = cv2.approxPolyDP(cont, 0.02 * peri, True)  # 多边形近似，得到多边形角点
            # approx = cv2.approxPolyDP(cont, 50, True)  # 多边形近似，得到多边形角点
            if len(approx) > 3:  # 如果大于三角形
                try: # TODO 部分图像报如下错误：ValueError: cannot reshape array of size 12 into shape (4,2)，
                    pts = approx.reshape((4, 2))  # 将角点位置转化为两列
                except:
                    continue
                sum = pts.sum(1)
                # print("pts:", pts) #[[ 371  160][ 407  537][1808  589][1914  284]]
                # print("sum:",sum) # [ 531  944 2397 2198]
                new_point = np.zeros((4, 1, 2), np.int32)


                new_point[0] = pts[np.argmin(sum)] # 最小点为所有角点中x+y最小值
                new_point[3] = pts[np.argmax(sum)] # 最大点为所有角点中x+y最大值
                diff = np.diff(pts, axis=1)
                new_point[1] = pts[np.argmin(diff)]
                new_point[2] = pts[np.argmax(diff)]

                # 偏移位置，扩大图像面积
                offset = np.array([[[-20, -20]], [[20, -20]], [[-20, 20]], [[20, 20]]])
                new_point = new_point + offset

                # print("new point", new_point)

                # cv2.drawContours(img, new_point, -1, (0,255,0),5) # 在目标角上打点
                points.append(new_point)
    return points  # 返回多张四边形点位的列表[嵌套array列表]



def get_wrap(img, biggest):
    '''
    视角变换
    :param img: 原始图像
    :param biggest: 四边形点位
    :return: 变换后的图像
    '''
    pts1 = np.float32(biggest) # 视角变换初始位置
    # print("point1:", pts1)
    pts2 = np.float32([[0, 0], [IMAGEWIDTH, 0], [0, IMAGEHEIGHT], [IMAGEWIDTH, IMAGEHEIGHT]]) # 视角变换最终位置，一般为图像大小
    # print("point2:", pts2)
    matrix = cv2.getPerspectiveTransform(pts1, pts2) # 设定视角变换矩阵
    imgOutput = cv2.warpPerspective(img, matrix, (IMAGEWIDTH, IMAGEHEIGHT)) # 输出后的图像
    # imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20] # 裁剪图像
    imgCropped = cv2.resize(imgOutput, (IMAGEWIDTH, IMAGEHEIGHT)) # 处理后的图像
    return imgCropped


def high_pass(img):
    '''
    高反差保留算法
    :param img: 输入图像
    :return: 输出图像-高反差保留
    '''
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    highPass = gray.astype(int) - blur.astype(int) + 128  # 高反差即为原始图减高斯模糊 +128为了调整亮度
    return highPass


def oswf(path):
    """
    :param path: 定义需要遍历的目录路径
    :return: 迭代器，返回当前目录下所有的文件及子文件
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            yield os.path.join(root, file)
        # for dir in dirs:
        #     oswf(os.path.join(root, dir))

if __name__ == '__main__':
    path = "./05-14"
    for f in oswf(path):
        if f.endswith(".jpg"): # 如果是jpg图像，进行处理，否则跳过
            f_basename = os.path.basename(f)
            # print(f_basename.split("-"))
            try:
                label = f_basename.split("-")[2]
            except:
                label = "05"
            if label in ["01", "03"]:
                IMAGEWIDTH = 1200
                IMAGEHEIGHT = 240
                trans = False
            elif label in ["02", "04"]:
                IMAGEWIDTH = 240
                IMAGEHEIGHT = 1200
                trans = True
            else:
                trans = False
                continue

            print("[Handling!!!]", f)
            img = cv2.imread(f)
            x, y = img.shape[0:2]
            # print("image size:",x,y)
            img_resize = cv2.resize(img, (int(y / 2), (int(x / 2))))
            # print("image size:",img_resize.shape[0:2])
            img_result = pre_process(img_resize)
            points = get_coutours(img_result)
            # print("points:",points)
            print("len(points):", len(points))
            # if len(points) == 4: # 如果points长度为4，也就是4个矩形框
            #     points = points
            # else: # 否则的话，为了所有的矩形框都能处理，根据前四张图手动设定，可不开
            #     if label == "01":
            #         points = [np.array([[[ 407,  962]],[[1704,  945]],[[ 417, 1244]],[[1631, 1197]]]),
            #                   np.array([[[416, 961]],[[1708, 941]],[[427, 1240]],[[1635, 1190]]]),
            #                   np.array([[[ 388,  597]],[[1812,  635]],[[ 402,  928]],[[1724,  928]]]),
            #                   np.array([[[ 351,  140]],[[1934,  264]],[[ 387,  557]],[[1828,  609]]])
            #                   ]
            #     elif label == "02":
            #         points = [np.array([[[1635,  300]],[[1969,  320]],[[1331, 1397]],[[1591, 1380]]]),
            #                   np.array([[[1234,  258]],[[1596,  280]],[[1037, 1412]],[[1317, 1393]]]),
            #                   np.array([[[ 789,  225]],[[1193,  252]],[[ 716, 1436]],[[1015, 1418]]]),
            #                   np.array([[[ 291,  190]],[[ 739,  227]],[[ 369, 1456]],[[ 688, 1442]]])
            #                   ]
            #     elif label == "03":
            #         points = [np.array([[[ 426, 1170]],[[1600, 1125]],[[ 431, 1416]],[[1536, 1348]]]),
            #                   np.array([[[ 406,  869]],[[1686,  866]],[[ 412, 1157]],[[1611, 1121]]]),
            #                   np.array([[[ 382,  493]],[[1788,  554]],[[ 393,  839]],[[1698,  848]]]),
            #                   np.array([[[ 362,   26]],[[1921,  179]],[[ 385,  456]],[[1812,  525]]])
            #                   ]
            #     elif label == "04":
            #         points = [np.array([[[1635,  222]],[[1959,  252]],[[1326, 1339]],[[1584, 1323]]]),
            #                   np.array([[[1246,  188]],[[1601,  220]],[[1036, 1358]],[[1312, 1343]]]),
            #                   np.array([[[ 806,  147]],[[1199,  180]],[[ 722, 1372]],[[1016, 1358]]]),
            #                   np.array([[[ 318,   86]],[[ 756,  129]],[[ 376, 1375]],[[ 693, 1378]]])
            #                   ]
            i = 1
            for p in points:
                img = get_wrap(img_resize, p)
                img = high_pass(img) # 高反差处理
                if trans:
                    img = cv2.transpose(img) # 如果是2,4工位的图像,则进行转置
                new_name_list = os.path.split(f) # 准备开始保存修改后的图片
                f_dirname, f_basename = new_name_list
                # print(f_basename)
                f_newname = f_basename.replace(".jpg", "-0" + str(i) + ".jpg")
                save_path = os.path.join(f_dirname,"highpass",f_newname)
                if not os.path.exists(os.path.dirname(save_path)):
                    os.mkdir(os.path.dirname(save_path))
                print("file saved:", save_path)
                cv2.imwrite(save_path, img, [int( cv2.IMWRITE_JPEG_QUALITY), 95]) # 保存图片
                i += 1

