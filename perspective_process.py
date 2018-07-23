import cv2
import numpy as np
import matplotlib.pyplot as plt
from process import draw_boxes, file_name

import os

filenames_ = file_name("img_datas")
print(filenames_)

sharpen = np.array(([-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]), dtype='int32')

# 第一种方法
def draw_contours(img_path, name='image'):

    img = cv2.imread(img_path, 0)
    blur_image = False
    img_backup = img.copy()
    print('shape:', img.shape)

    imageVar = cv2.Laplacian(img, cv2.CV_64F).var()
    print(imageVar)
    if imageVar < 250:
        blur_image = True
        name = 'BLUR--'+str(imageVar)+'--'+name
        img = cv2.filter2D(img, -1, sharpen)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=4)
    ###########################################################################
    # 寻找最小区域
    ###########################################################################

    def find_contours(img_, draw_=True, k=7):
        kernel = np.ones((k, k), np.uint8)
        binary = cv2.adaptiveThreshold(img_, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, k, 13)
        # cv2.imshow('{}_0'.format(name), binary)
        binary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        # binary = cv2.medianBlur(binary, k)
        binary = cv2.GaussianBlur(binary, (3, 3), 0)  # 改用高斯模糊

        # cv2.imshow('{}_1'.format(name), binary)
        image, contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if draw_:
            boxes = []
            for c in contours:
                # area_ = cv2.contourArea(c)
                # if area_ / (img_.shape[0] * img_.shape[1]) < 0.0015:
                #     continue
                rect = cv2.minAreaRect(c)
                # angle = rect[-1]
                # print('angle', angle)
                box = cv2.boxPoints(rect)
                box = np.int_(box)
                boxes.append(box)
                # print(box)
                # cv2.drawContours(img_, [box], -1, (0, 0, 255), 1)  # 把小框画到图片上 用于第二轮
            return boxes
        else:
            # 第二轮 用于寻找最大的框
            print(len(contours))
            max_area = 0
            max_contour = contours[0]
            for c in contours:
                area_ = cv2.contourArea(c)
                if area_ > max_area:
                    max_area = area_
                    max_contour = c

            # 如果最大框占比 小于 百分之二十五
            print('max_area rates', max_area/(img_.shape[0]*img_.shape[1]))
            if max_area/(img_.shape[0]*img_.shape[1]) < 0.2:
                return []

            # 如果最大框占比 大于 百分之二十五
            rect_ = cv2.minAreaRect(max_contour)
            box = cv2.boxPoints(rect_)
            # print(box)
            box = np.int_(box)
            cv2.drawContours(img_, [box], -1, (0, 0, 255), 1)
            print(box)
            return box
            # return contours
        # return img_

    bs = find_contours(img, k=3)  # 画小框 第一轮
    # print(bs)
    if blur_image:
        # img = img_backup.copy()
        for box in bs:
            cv2.drawContours(img_backup, [box], -1, (0, 0, 255), 1)
        cv2.imshow('{}_small_box'.format(name), img_backup)
    else:
        for box in bs:
            cv2.drawContours(img_backup, [box], -1, (0, 0, 255), 1)
        cv2.imshow('{}_small_box'.format(name), img_backup)
    # c_ = find_contours(img, k=7, draw_=False)  # 画大框  第二轮
    # cv2.imshow('first_1', img)

    # return c_


#  旋转图片 暂时没用到
def rotate_image(img_path, name='rotate image'):
    img = cv2.imread(img_path, 0)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(img)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow('first_', thresh)
    # print(thresh.shape)
    coords = np.column_stack(np.where(thresh > 0))
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    print('angle and rect', -angle, rect)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # print('angle ', angle)
    (h, w) = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # cv2.imshow('rotated_', rotated)
    return rotated
    # print(img.shape)


# 放大图像
# fx = 1.001
# fy = 1.001
# _imgs = cv2.resize(_imgs, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
# print(_imgs.shape)

# 综合两种方法画框
def detect_box(path_):
    print('###################################################')
    p = draw_contours(path_, name='one')
    # 如果第一种方法没有结果  直接使用旧方法 或者 旋转图片的方法
    if len(p) == 0:
        print('rotate')
        print('###################################################')
        # rot = rotate_image(path_)
        img = cv2.imread(path_, 0)
        # cv2.imshow('rotated_', rot)
        draw_boxes(img)
        return []

    # 如果第一种方法有结果
    print('perspective')
    print('###################################################')
    _imgs = cv2.imread(path, 0)
    # 获取图片的行列
    h_max = _imgs.shape[0]
    w_max = _imgs.shape[1]
    # 获取图片的中点
    h_mid = _imgs.shape[0]//2
    w_mid = _imgs.shape[1]//2

    real_x = np.max(p[:, 0])
    real_y = np.max(p[:, 1])

    if real_y <h_mid:
        h_max = h_mid
        h_mid = real_y // 2

    if real_x <w_mid:
        w_max = w_mid
        w_mid = real_x // 2

    pts1 = np.float32(p)

    # 计算目标坐标
    pts_tmp = []
    for i in pts1:
        i0, i1 = i
        if i0 > w_mid and i1 > h_mid:
            pts_tmp.append([w_max, h_max])
        elif i0 < w_mid and i1 > h_mid:
            pts_tmp.append([0, h_max])
        elif i0 < w_mid and i1 < h_mid:
            pts_tmp.append([0, 0])
        elif i0 > w_mid and i1 < h_mid:
            pts_tmp.append([w_max, 0])

    print(pts_tmp)
    pts2 = np.float32(pts_tmp)

    # 透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    res = cv2.warpPerspective(_imgs, M, (w_max, h_max), borderMode=1)

    cv2.drawContours(_imgs, [p], -1, (0, 0, 255), 1)

    cv2.imshow('image and area', _imgs)
    cv2.imshow('perspective image', res)
    draw_boxes(res, name='pers')


###########################################################################
# 使用
###########################################################################


path = 'img_datas/352584T20170925075215.png'
path_header = 'img_datas/'

draw_contours(path)

# for i in filenames_:
#     draw_contours(path_header+i, name=i)

cv2.waitKey()
cv2.destroyAllWindows()


