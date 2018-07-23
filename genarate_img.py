#-*- coding:utf8 -*-

import sys, random, os
import numpy as np
from captcha.image import ImageCaptcha
from PIL import Image
from PIL import ImageFont, ImageFilter
from PIL.ImageDraw import Draw
import cv2
import utils
from multiprocessing import Pool
from math import *
import re

file_name = 'ch_test'

imgDir = file_name
numProcess = 12


# 生产随机数
def r(val):
    return int(np.random.random() * val)


# 图片旋转
def rotRandom(img, factor, size, background):
    shape = size
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [r(factor), shape[0] - r(factor)], [shape[1] - r(factor), r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size,  borderValue=background)
    return dst


def rot(img, angel, shape, max_angel):
    size_o = [shape[1], shape[0]]  # h, w

    # 角度 转 弧度
    size = (shape[1] + int(shape[0] * cos((float(max_angel) / 180) * 3.14)), shape[0])
    interval = abs(int(sin((float(angel) / 180) * 3.14) * shape[0]))

    pts1 = np.float32([[0, 0], [0, size_o[1]], [size_o[0], 0], [size_o[0], size_o[1]]])
    if angel > 0:
        pts2 = np.float32([[interval, 0], [0, size[1]], [size[0], 0], [size[0] - interval, size_o[1]]])
    else:
        pts2 = np.float32([[0, 0], [interval, size[1]], [size[0] - interval, 0], [size[0], size_o[1]]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, size, borderMode=cv2.BORDER_REPLICATE, borderValue=0)

    return dst


# 随机生产颜色
def random_color(start, end, opacity=None):
    seed = random.randint(start, end)
    red = seed
    green = seed
    blue = seed - random.randint(0, 16)
    if opacity is None:
        return red, green, blue
    return red, green, blue, opacity


###############################################################################
# 生产图片的类
###############################################################################

class ImageCaptcha_(ImageCaptcha):

    def create_captcha_image(self, chars, color, background):
        image = Image.new('RGB', (self._width, self._height), background)

        def _bg_noise(image_, i=None):
            if i is None:
                i = [-10, 10]

            image_ = np.array(image_, dtype='int16')
            raw_user_item_mat = np.random.randint(i[0], i[1], size=image_.shape, dtype='int16')
            image_ += raw_user_item_mat  # background
            image_ = Image.fromarray(np.uint8(image_))
            return image_

        def create_img(c_, font_, size_, background_, ink_):
            dx, dy = 1, 1
            font_ = ImageFont.truetype(font_, size_)  # 中文字体
            w, h = draw.textsize(c_, font=font_)
            im = Image.new('RGB', (w + dx * 1, h + dy * 1), background_)
            # im = _bg_noise(im)
            # 文字颜色
            Draw(im).text((dx, dy), c_, font=font_,
                          fill=(random.randint(ink_, ink_), random.randint(ink_, ink_),
                                random.randint(ink_, ink_)))
            return im

        # image = _bg_noise(image)
        draw = Draw(image)
        s = random.randint(28, 28)

        ch_fonts = ['./fonts/sb.ttf']
        random_ch_font = random.choice(ch_fonts)

        num_eng_fonts = ['./fonts/times.ttf', './fonts/timesi.ttf', './fonts/cambria.ttc', './fonts/cambriai.ttf']
        random_font = random.choice(num_eng_fonts)
        ink_num = random.choice([40, 50, 60, 70])

        # rota = random.uniform(0, 0)
        def _draw_character(c, size, rotate=None):
            if c not in list(utils.char_ + utils.punctuation):
                im = create_img(c, random_ch_font, size, background, ink_num)
            else:
                if c in list(utils.punctuation):
                    im = create_img(c, random_ch_font, size, background, ink_num)
                else:
                    im = create_img(c, random_font, size+random.randint(2, 3), background, ink_num)

            return im

        images = []
        # b = 0
        for c in chars:
            # blank_ = random.randint(1, 5)
            # if len(chars) < 6 and blank_ == 1 and b < 1:
            #     b += 1
            #     im = _draw_character('  ', s)
            #     images.append(im)

            im = _draw_character(c, s)
            images.append(im)

        text_width = sum([im.size[0] for im in images])
        average = int(text_width / len(chars))

        # 增加空格
        blank_ = random.randint(1, 3)
        if len(chars) < 8 and blank_ == 1:
            offset = average*random.randint(0, 2) + average//3
        else:
            offset = average // 3

        for im in images:
            w, h = im.size
            image.paste(im, (offset, int((self._height - h) / 2)))
            offset = offset + w

        # 噪音 曲线
        color_noise = random_color(1, 3)
        noise_exist = random.randint(1, 3)
        if noise_exist == 1:
            self.create_noise_dots(image, color_noise, width=random.randint(3, 6), number=random.randint(5, 8))

        curve_exist = random.randint(1, 4)
        if curve_exist == 1 and len(chars) > 1:
            self.create_noise_curve(image, color_noise)

        image = image.filter(ImageFilter.DETAIL)

        # 转灰度
        image = image.convert('L')

        # 转成numpy
        image = np.array(image, dtype='int16')

        # image = rot(image, r(30) - 15, image.shape, 15)
        # image = cv2.resize(image, (self._width, self._height))
        # print(image.shape)
        # 图片旋转
        image = rotRandom(image, 5, (image.shape[1], image.shape[0]), background)
        image = rotRandom(image, 3, (image.shape[1], image.shape[0]), background)

        image = Image.fromarray(np.uint8(image))
        image = image.filter(ImageFilter.SHARPEN)

        # 模糊
        sm = random.randint(1, 6)
        if sm == 1:
            image = image.filter(ImageFilter.SMOOTH_MORE)
            image = image.filter(ImageFilter.SMOOTH_MORE)
            image = image.filter(ImageFilter.SMOOTH_MORE)
        elif sm == 4 or 5:
            image = image.filter(ImageFilter.SMOOTH_MORE)
            image = image.filter(ImageFilter.SMOOTH_MORE)
        elif sm == 3:
            image = image.filter(ImageFilter.SMOOTH)
        else:
            image = image.filter(ImageFilter.SMOOTH)
            image = image.filter(ImageFilter.SMOOTH_MORE)

        # 背景模糊
        image = _bg_noise(image, i=[-5, 5])

        return image

    def generate_image(self, chars):
        # background_c = random.choice([120, 130, 140, 150, 160, 170, 180, 190])
        background = random_color(130, 200)
        color = random_color(0, 4)
        im = self.create_captcha_image(chars, color, background)

        return im

def gen_rand(ind=None):
    buf = []
    if ind is not None:
        buf.append(utils.charset[ind % len(utils.charset)])
        for i in range(random.randint(1, 4)):
            buf.append(random.choice(utils.char_))  # 添加英文、数字
        random.shuffle(buf)

        return ''.join(buf)

    max_len = random.randint(1, 5)
    for i in range(max_len):
        buf += random.choice(utils.charset)
    return ''.join(buf)

# 随机生成图片内的文字
def gen_rand2(ind=None):
    l_math = len(utils.math_only)
    seq = utils.math_only[ind % l_math]

    return seq


# 随机生成图片内的文字
def gen_rand3(ind=None):
    buf = []
    if ind is not None:
        buf.append(utils.charset[ind % len(utils.charset)])
        tmp = random.randint(0, 5)

        # 添加中文或英文、数字
        for i in range(random.randint(0, 10)):
            if tmp == 5:
                buf.append(random.choice(utils.char_))  # 添加英文、数字

            elif tmp == 4:
                if i < tmp:  # 部分中文  部分英文数字
                    buf.append(random.choice(utils.char_ + utils.punctuation))
                elif i >= 4 and len(buf) < 8:
                    buf.append(random.choice(utils.word_1))

            else:  # 添加中文或英文、数字
                break

        if tmp < 4:
            buf.append(utils.word_1[ind % len(utils.word_1)])
            buf.append(utils.word_2[ind % len(utils.word_2)])
            # buf.append(random.choice(utils.word_2))

        random.shuffle(buf)

        return ''.join(buf)

    max_len = random.randint(1, 4)
    for i in range(max_len):
        buf += random.choice(utils.charset)
    return buf


# ==============================================================================
# 以下方法训练时不会调用
# ==============================================================================

# 生成图片
def generate_img(ind):
    global imgDir
    # for i in range(ind):
    captcha = ImageCaptcha_(width=256, height=48)
    theChars = gen_rand(ind)
    # data = captcha.generate(theChars)
    # print(theChars)
    img_name = '{:08d}'.format(ind)+'_'+theChars+'.png'
    img_path = imgDir+'/'+img_name
    captcha.write(theChars, img_path)  # 调用generate_image
    print(img_path)


def run(num, path):
    global imgDir
    imgDir = path
    if not os.path.exists(path):
        os.mkdir(path)
    # generate_img(num, path)
    with Pool(processes=numProcess) as pool:
         pool.map(generate_img, range(num))


if __name__ == '__main__':

    # run(64000, file_name)

    run(1, file_name)

    # wordl_ = open('haha_2.txt','w',encoding='utf-8')
    # with open('haha.txt', encoding='utf-8') as f:
    #     xx = 0
    #     for i in f:
    #         print(xx)
    #         xx+=1
    #         i = i.strip()
    #         i = i.encode('utf-8').decode('utf-8-sig')
    #         i = [i_ for i_ in i if i_ in utils.charset_list]
    #         i = ''.join(i)
    #         wordl_.write(i+'\n')
    # wordl_.close()
