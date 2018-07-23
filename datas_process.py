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

filenames_ = file_name("img_datas")
print(filenames_)


img = cv2.imread('img_datas/22195778T20170925094654.png', 0)

kernel = np.ones((3, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)

# # plt.figure()
# plt.subplot(2, 3, 1)
# plt.imshow(img, 'gray')  # 默认彩色，另一种彩色bgr
# plt.subplot(2, 3, 2)
# plt.imshow(erosion, 'gray')
#
# plt.subplot(2, 3, 3)
# plt.imshow(dilation, 'gray')
#
# # 先进性腐蚀再进行膨胀就叫做开运算。就像我们上面介绍的那样，它被用来去除噪声。这里我们用到的函数是cv2.morphologyEx()
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#
# # 先膨胀再腐蚀。它经常被用来填充前景物体中的小洞，或者前景物体上的小黑点
# closing = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# ret, thresh4 = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)
#
# plt.subplot(2, 3, 4)
# plt.imshow(thresh4, 'gray')  # 默认彩色，另一种彩色bgr
# plt.subplot(2, 3, 5)
# plt.imshow(opening, 'gray')  # 默认彩色，另一种彩色bgr
# plt.subplot(2, 3, 6)
# plt.imshow(closing, 'gray')
#
# pylab.show()


# retval, im_at_fixed = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
# plt.axis("off")
# plt.title("Fixed Thresholding")
# plt.imshow(im_at_fixed, cmap = 'gray')
# pylab.show()

# closing = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# im_at_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 9)
# im_at_mean = cv2.morphologyEx(im_at_mean, cv2.MORPH_GRADIENT, kernel)
# plt.axis("off")
# plt.title("Adaptive Thresholding with mean weighted average")
# plt.imshow(im_at_mean, cmap='gray')
# pylab.show()


#
# im_at_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 9)
# plt.axis("off")
# plt.title("Adaptive Thresholding with gaussian weighted average")
# plt.imshow(im_at_mean, cmap='gray')
# pylab.show()

# gray = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
# edges = cv2.Canny(gray, 50, 120)
# edges = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
# cv2.imshow("edges", edges)



binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 13)

# binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
cv2.imshow("img0", binary)
binary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
binary = cv2.medianBlur(binary,3)
# binary = cv2.blur(binary,(3,3))
# binary = cv2.adaptiveThreshold(binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 11)
cv2.imshow("img1", binary)
image, contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(np.array(contours).shape)
# cv2.drawContours(img, contours[1], -1, (0, 0, 255), 3)
w_l = []
h_l = []

for c in contours:
    # print(np.array(c).shape)
    x,y,w,h = cv2.boundingRect(c)

    # if w23 and h
    # if w<12 or h<13:
    #     continue
    print(x, y, w, h)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    w_l.append(w)
    h_l.append(h)
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

print(np.mean(w_l),'——',w_l)
print(np.mean(h_l),'——',h_l)


cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
# cv2.imshow("binary2", binary)
# im_at_mean = cv2.morphologyEx(im_at_mean, cv2.MORPH_OPEN, kernel)
# im_at_mean = cv2.morphologyEx(im_at_mean, cv2.MORPH_CLOSE, kernel)
# im_at_mean = cv2.morphologyEx(im_at_mean, cv2.MORPH_OPEN, kernel)
# cv2.imshow("Adaptive Thresholding with gaussian weighted average", im_at_mean)
# im_at_mean = cv2.Canny(im_at_mean, 50, 120)
# im_at_mean = cv2.morphologyEx(im_at_mean, cv2.MORPH_GRADIENT, kernel)
# cv2.imshow("Adaptive Thresholding with gaussian weighted average______", im_at_mean)


# minLineLength = 100
# maxLineGap = 5
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 180, minLineLength, maxLineGap)
# print(lines)
# for i in lines:
#     for x1,y1,x2,y2 in i:
#         cv2.line(img, (x1,y1),(x2,y2),(0,255,0),2)
#
# cv2.imshow("edges", edges)
# cv2.imshow("lines", img)
cv2.waitKey()
cv2.destroyAllWindows()


# edges = cv2.Canny(img, 50, 150, apertureSize=3)
# lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
# result = img.copy()
# for i in lines:
#     for line in i:
#         rho = line[0]
#         theta = line[1]
#         if(theta < (np.pi/4. )) or (theta > (3.*np.pi/4.0)):
#             pt1 = (int(rho/np.cos(theta)),0)
#             pt2 = (int((rho-result.shape[0]*np.sin(theta))/np.cos(theta)),result.shape[0])
#             cv2.line( result, pt1, pt2, (0,0,255))
#         else:
#             pt1 = (0,int(rho/np.sin(theta)))
#             pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
#             cv2.line(result, pt1, pt2, (0,0,255), 1)
# cv2.imshow('Hough', result)
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()
