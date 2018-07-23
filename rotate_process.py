import cv2
import numpy as np
import matplotlib.pyplot as plt
from process import draw_boxes

sharpen = np.array(([-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]), dtype='int32')


# 第一种方法
def draw_contours(img_path, name='image'):

    img = cv2.imread(img_path, 0)
    img_backup = img.copy()
    print('shape:', img.shape)

    # imageVar = cv2.Laplacian(img, cv2.CV_64F).var()
    # print(imageVar)
    #
    # blur_image = False
    # if imageVar < 250:
    #     blur_image = True
    #     name = 'BLUR--' + str(imageVar) + '--' + name
    #     img = cv2.filter2D(img, -1, sharpen)
    #     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=4)

    emptyImage = img.copy()
    fx = 1.1
    fy = 1.1
    emptyImage = cv2.resize(emptyImage, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    print('shape:', emptyImage.shape, np.mean(img))
    emptyImage[...] = np.mean(img)
    # cv2.imshow('empty_0', emptyImage)

    ###########################################################################
    # 寻找最小区域
    ###########################################################################

    def find_contours(img_, draw_=True, k=7):
        kernel = np.ones((k, k), np.uint8)
        binary = cv2.adaptiveThreshold(img_, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, k, 13)
        # cv2.imshow('{}_0'.format(name), binary)
        binary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
        # binary = cv2.medianBlur(binary, k)
        binary = cv2.GaussianBlur(binary, (5, 5), 0)  # 改用高斯模糊

        # cv2.imshow('{}_1'.format(name), binary)
        image, contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if draw_:
            cp = 0
            points_old = []
            for c in contours:
                rect_ = cv2.minAreaRect(c)
                points_old.append(rect_)  # 把所有rect取出来

            print(points_old)
            points_old.sort(key=lambda x: x[0][1])
            print(points_old)
            # bs = []
            for rect in points_old:
                cp += 1
                print(cp, '########################################')
                angle = rect[-1]
                center = rect[0]
                w_h = np.int_(rect[1])
                # print('angle and rect', -angle, rect)
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                # print('angle ', angle)

                box = cv2.boxPoints(rect)
                # print(type(box))
                # box = np.int_(box)
                box_ = np.int_(box)
                pts1 = np.float32(np.int_(box))

                if abs(angle) != 0:
                    if angle > 0:
                        print('angle bigger than zero')
                        (h_, w_) = w_h[1], w_h[0]
                        h_mid = center[1]
                        w_mid = center[0]

                    else:
                        print('angle less than zero')
                        (h_, w_) = w_h[0], w_h[1]
                        h_mid = center[1]
                        w_mid = center[0]

                    # 计算目标坐标
                    pts_tmp = []
                    for i in box:
                        i0, i1 = i  # print(i0, i1, w_mid, h_mid)
                        if i0 > w_mid and i1 > h_mid and [w_, h_] not in pts_tmp:
                            pts_tmp.append([w_, h_])
                        elif i0 < w_mid and i1 > h_mid and [0, h_] not in pts_tmp:
                            pts_tmp.append([0, h_])
                        elif i0 < w_mid and i1 < h_mid and [0, 0] not in pts_tmp:
                            pts_tmp.append([0, 0])
                        elif i0 > w_mid and i1 < h_mid and [w_, 0] not in pts_tmp:
                            pts_tmp.append([w_, 0])

                    if len(pts_tmp) < 4:
                        print('asdadsasdadad', box, h_mid, w_mid)
                        box[box[:, 0] < w_mid, 0] = 0
                        box[box[:, 0] > w_mid, 0] = w_

                        a, b = box[box[:, 0] == 0, 1]
                        if a > b:
                            box[box[:, 0] == 0, 1] = h_, 0
                        else:
                            box[box[:, 0] == 0, 1] = 0, h_

                        a, b = box[box[:, 0] == w_, 1]
                        if a > b:
                            box[box[:, 0] == w_, 1] = h_, 0
                        else:
                            box[box[:, 0] == w_, 1] = 0, h_

                        pts_tmp = box

                    # print(h_, w_, center, box)

                    # box = np.int_(box)
                    # pts1 = np.float32(box)

                    print(pts1, pts_tmp)
                    pts2 = np.float32(pts_tmp)

                    # 透视变换
                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    res = cv2.warpPerspective(img_, M, (w_, h_), borderMode=1)
                    cv2.imshow('asdad_{}'.format(cp), res)

                    # print(res.shape)
                    x1, y1 = (int(w_mid) - res.shape[1]//2, int(h_mid) - res.shape[0]//2)  # x,y
                    if y1 < 0:
                        y1 = 0
                    if x1 < 0:
                        x1 = 0
                    print('rect points and w h', x1, y1, res.shape[1], res.shape[0])

                    emptyImage[y1:y1 + h_, x1:x1 + w_] = res
                else:
                    box = np.int_(box)
                    xs = list(set([i[0] for i in box]))
                    ys = list(set([i[1] for i in box]))

                    w_ = abs(xs[0] - xs[1])
                    h_ = abs(ys[0] - ys[1])
                    x1 = min(xs)
                    y1 = min(ys)

                    print('points', x1, y1, w_, h_)
                    emptyImage[y1:y1 + h_, x1:x1 + w_] = img_[y1:y1 + h_, x1:x1 + w_]

                # area_ = cv2.contourArea(c)
                # if area_ / (img_.shape[0] * img_.shape[1]) < 0.001:
                #     continue
                # print(box)
                cv2.drawContours(img_, [box_], -1, (0, 0, 255), 1)  # 把小框画到图片上 用于第二轮
                # if cp == 8:
                #     break

    find_contours(img, k=3)  # 画小框 第一轮
    cv2.imshow('first_0', img)
    cv2.imshow('empty_1', emptyImage)



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
    h_max = _imgs.shape[0]
    w_max = _imgs.shape[1]

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


path = 'img_datas/350894T20170925155929.png'

draw_contours(path)

cv2.waitKey()
cv2.destroyAllWindows()


