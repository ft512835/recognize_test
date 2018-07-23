import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pylab

#  获取当前文件夹下的所有文件（不包括子文件夹）
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        # print(dirs)  # 当前路径下所有子目录nim
        # print(files)  # 当前路径下所有非目录子文件
        # files = [root+"/"+i for i in files]
        return files

# filenames_ = file_name("img_datas")
# print(filenames_)

# img_ = cv2.imread('img_datas/22195776T20170925063603.png', 0)


def find_points(points):
    def sort_y(_x):
        return _x[1]

    h_l = [i[-1] for i in points]
    points.sort(key=sort_y)
    print('point:', points)  # 按y排序的坐标

    hls = sorted(h_l)
    h_mean = np.mean(hls)
    h_mid = hls[int(len(hls) * 3 / 5)]  # 高度中位数
    hm = h_mean
    hm_ = min([hm, (h_mid + h_mean) / 2])

    print('hm_,hm:', hm_, hm)

    print('###################################################')
    points_ = []
    group = []
    for i, _ in enumerate(points):
        if i + 1 == len(points):
            group.append(points[i])
            break

        # x, y, w, h = points[i]
        x2, y2, w2, h2 = points[i + 1]

        if not group:
            group.append(points[i])

        y_min = min([i[1] for i in group])

        if abs(y2 - y_min) < np.int_(hm_):
            group.append(points[i + 1])
            # print(group)
        else:
            if len(group) == 1:
                print('group_1：', group)
                if group[0][3] <= np.ceil(hm_ * 2 / 3) or group[0][3] > 2.5 * hm_:
                    group = []
                    continue
                points_.append(group[0])
                group = []
            else:
                print('group_2：', group)
                x_ = []
                y_ = []
                w_fake = []
                h_fake = []
                for j in group:
                    if j[3] < np.int_(hm_ / 3):
                        pass
                        # j[3] = np.int_(hm_*2/3)
                        # elif hm_ * 3 > j[3] > np.int_(hm_ + hm_/3):
                        # j[3] = np.int_(hm_+hm_/2)
                    elif j[3] > hm_ * 2.5:
                        continue
                    x_.append(j[0])
                    y_.append(j[1])
                    w_fake.append(j[0] + j[2])
                    h_fake.append(j[1] + j[3] - y_min)

                if len(h_fake) == 0:
                    group = []
                    continue

                h_real = int(max(h_fake))
                if h_real <= np.ceil(hm_ * 2 / 3):
                    group = []
                    continue
                # h_real += np.int_(hm_/3)
                x_min = min(x_)
                y_min = min(y_)
                w_real = max(w_fake)
                print(x_min, y_min, w_real, h_real)
                points_.append([x_min, y_min, w_real, h_real])
                group = []

    # 如果group仍有值
    if group:
        print('group_3：', group)
        y_min = min([i[1] for i in group])
        x_ = []
        y_ = []
        w_fake = []
        h_fake = []
        for j in group:
            if j[3] < np.int_(hm_ / 3):
                pass
                # j[3] = np.int_(hm_ * 2 / 3)
            # elif hm_ * 3 > j[3] > np.int_(hm_ + hm_ / 2):
            #     j[3] = np.int_(hm_ + hm_ / 2)
            elif j[3] > hm_ * 2.5:
                continue
            x_.append(j[0])
            y_.append(j[1])
            w_fake.append(j[0] + j[2])
            h_fake.append(j[1] + j[3] - y_min)

        h_real = int(max(h_fake))

        x_min = min(x_)
        y_min = min(y_)
        w_real = max(w_fake)
        print(x_min, y_min, w_real, h_real)
        if h_real > hm_:
            points_.append([x_min, y_min, w_real, h_real])
            # group = []
    return points_

def draw_boxes(img, name='boxes', k=5):
    kernel = np.ones((3, 5), np.uint8)
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 13)
    # cv2.imshow("{}_0".format(name), binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    # binary = cv2.medianBlur(binary, 5)
    binary = cv2.GaussianBlur(binary, (k, k), 0)

    cv2.imshow("{}_1".format(name), binary)
    image, contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(np.array(contours).shape)


    # 处理边框属性
    # w_l = []
    h_l = []
    points = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # w_l.append(w)
        h_l.append(h)
        points.append([x, y, w, h])

        # rect = cv2.minAreaRect(c)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # print(box)
        # cv2.drawContours(img, [box], 0, (0,0,255), 4)

        # (x,y), radius = cv2.minEnclosingCircle(c)
        # center = (int(x), int(y))
        # radius = int(radius)
        # img = cv2.circle(img, center, radius, (0,255,0),2)
    # cv2.drawContours(img, contours, -1, (255,0,0), 1)

##################################################
    # def sort_y(_x):
    #     return _x[1]
    #
    # points.sort(key=sort_y)
    # print('point:', points)  # 按y排序的坐标
    #
    # hls = sorted(h_l)
    # h_mean = np.mean(hls)
    # h_mid = hls[int(len(hls)*3/5)]   # 高度中位数
    # hm = h_mean
    # hm_ = min([hm, (h_mid+h_mean)/2])
    #
    # print('hm_,hm:', hm_, hm)
    #
    # print('###################################################')
    # points_ = []
    # group = []
    # for i, _ in enumerate(points):
    #     if i+1 == len(points):
    #         break
    #
    #     # x, y, w, h = points[i]
    #     x2, y2, w2, h2 = points[i+1]
    #
    #     if not group:
    #         group.append(points[i])
    #
    #     y_min = min([i[1] for i in group])
    #
    #     if abs(y2-y_min) < np.int_(hm_):
    #         group.append(points[i+1])
    #         # print(group)
    #     else:
    #         if len(group) == 1:
    #             print('group_1：', group)
    #             if group[0][3] < hm_*2/3 or group[0][3] > 2.5*hm_:
    #                 group = []
    #                 continue
    #             points_.append(group[0])
    #             group = []
    #         else:
    #             print('group_2：', group)
    #             x_ = []
    #             y_ = []
    #             w_fake = []
    #             h_fake = []
    #             for j in group:
    #                 if j[3] < np.int_(hm_/3):
    #                     pass
    #                     # j[3] = np.int_(hm_*2/3)
    #                 # elif hm_ * 3 > j[3] > np.int_(hm_ + hm_/3):
    #                     # j[3] = np.int_(hm_+hm_/2)
    #                 elif j[3] > hm_ * 2.5:
    #                     continue
    #                 x_.append(j[0])
    #                 y_.append(j[1])
    #                 w_fake.append(j[0]+j[2])
    #                 h_fake.append(j[1]+j[3]-y_min)
    #
    #             if len(h_fake) == 0:
    #                 group = []
    #                 continue
    #
    #             h_real = int(max(h_fake))
    #             if h_real < hm_*2/3:
    #                 group = []
    #                 continue
    #             # h_real += np.int_(hm_/3)
    #             x_min = min(x_)
    #             y_min = min(y_)
    #             w_real = max(w_fake)
    #             print(x_min, y_min, w_real, h_real)
    #             points_.append([x_min, y_min, w_real, h_real])
    #             group = []
    #
    # # 如果group仍有值
    # if group:
    #     print('group_3：', group)
    #     y_min = min([i[1] for i in group])
    #     x_ = []
    #     y_ = []
    #     w_fake = []
    #     h_fake = []
    #     for j in group:
    #         if j[3] < np.int_(hm_ / 3):
    #             pass
    #             # j[3] = np.int_(hm_ * 2 / 3)
    #         # elif hm_ * 3 > j[3] > np.int_(hm_ + hm_ / 2):
    #         #     j[3] = np.int_(hm_ + hm_ / 2)
    #         elif j[3] > hm_ * 2.5:
    #             continue
    #         x_.append(j[0])
    #         y_.append(j[1])
    #         w_fake.append(j[0] + j[2])
    #         h_fake.append(j[1] + j[3] - y_min)
    #
    #     h_real = int(max(h_fake))
    #
    #     x_min = min(x_)
    #     y_min = min(y_)
    #     w_real = max(w_fake)
    #     print(x_min, y_min, w_real, h_real)
    #     if h_real > hm_:
    #         points_.append([x_min, y_min, w_real, h_real])
    #     # group = []
##################################################

    points_ = find_points(points)
    # points_ = find_points(points_)
    print('###################################################')
    # 画新的框
    print('points_：', points_)
    for i, _ in enumerate(points_):
        x, y, w, h = points_[i]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("{}_2".format(name), img)

    # 画旧的框
    for i, _ in enumerate(points):
        x, y, w, h = points[i]

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("{}_3".format(name), img)

path = 'img_datas/22195820T20170925055905.png'
img = cv2.imread(path, 0)
draw_boxes(img)

cv2.waitKey()
cv2.destroyAllWindows()
# g = open('words_.txt','w', encoding='utf-8')
# with open('words.txt', encoding='utf-8') as f:
#     for i_ in f:
#         i_ = i_.strip().split()[0]
#         g.write(i_+'\n')

# g.close()
