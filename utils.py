#-*- coding:utf8 -*-

import os, sys, re
#import config
import numpy as np
import tensorflow as tf
import random
import cv2, time
from genarate_img import ImageCaptcha_, gen_rand
from tensorflow.python.client import device_lib


channel = 1
image_width = 256
image_height = 48
num_features = image_height*channel
SPACE_INDEX = 0
SPACE_TOKEN = ''

maxPrintLen = 128

tf.app.flags.DEFINE_integer('image_width', image_width, 'image_width')
tf.app.flags.DEFINE_integer('image_height', image_height, 'image_height')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4, 'inital lr')

tf.app.flags.DEFINE_integer('num_layers', 2, 'number of layer')
tf.app.flags.DEFINE_integer('num_hidden', 512, 'number of hidden')
tf.app.flags.DEFINE_integer('num_epochs', 10000, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size', 4, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 1000, 'the step to save checkpoint')
tf.app.flags.DEFINE_integer('validation_steps', 1000, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 0.9, 'the lr decay rate')
tf.app.flags.DEFINE_integer('decay_steps', 50000, 'the lr decay_step for optimizer')

tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

FLAGS = tf.app.flags.FLAGS

# num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)


def get_chinese():
    f = open('chinese.txt', 'r', encoding='utf-8')
    chinese = f.read().strip()
    f.close()
    chinese = chinese.encode('utf-8').decode('utf-8-sig')
    return chinese


def get_chinese2():
    chinese = [chr(i) for i in range(ord('一'), ord('龥'))]
    return ''.join(chinese)

# ch_ = get_chinese2()
# ch = get_chinese()
ch = ''
punctuation = '？。，；、'
# ch = ''
char_ = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ()+-=.'
# char_set = char_ + punctuation + ch
charset = char_ + punctuation + ch
charset_list = list(charset)
# 26*2 + 10 digit + blank + space + chinese + bracket
# num_classes = 26+26+10+3756+3
num_classes = len(charset) + 2  # blank + space
# num_classes = 26+26+10+1+1+8

encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN


def read_words2():
    word1 = []
    with open('haha_.txt', encoding='utf-8') as f:
        for i in f:
            i = i.strip()
            i = i.encode('utf-8').decode('utf-8-sig')
            word1.append(i)
    return word1

# math_only = read_words2()


def read_words():
    word1 = []
    with open('words.txt', encoding='utf-8') as f:
        for i in f:
            i = i.strip()
            i = i.encode('utf-8').decode('utf-8-sig')
            word1.append(i)

    word2 = []
    with open('math.txt', encoding='utf-8') as f:
        for i in f:
            i = i.strip()
            i = i.encode('utf-8').decode('utf-8-sig')
            word2.append(i)

    return word1, word2

# word_1, word_2 = read_words()


class DataIterator:
    def __init__(self, data_dir=None):
        self.data_dir = None
        if data_dir is not None:
            self.data_dir = data_dir
            self.image_names = []
            self.image = []
            self.labels = []
            for root, sub_folder, file_list in os.walk(data_dir):
                for file_path in file_list:
                    image_name = os.path.join(root, file_path)
                    self.image_names.append(image_name)
                    code = image_name.split('\\')[1].split('_')[1]
                    code = re.split('\.png', code)[0]
                    code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]
                    self.labels.append(code)
        else:
            self.captcha = ImageCaptcha_(width=image_width, height=image_height)

    @property
    def size(self):
        if self.data_dir is None:
            return 6400000
        return len(self.labels)

    # 批量生成图片
    def generate_batch_img(self, ind=None, ind_list=None):
        label_batch = []
        image_batch = []
        if ind is not None:
            for i_ in range(ind):
                theChars = gen_rand(i_)
                code = [SPACE_INDEX if theChars == SPACE_TOKEN else encode_maps[c] for c in list(theChars)]
                label_batch.append(code)
                image = self.captcha.generate_image(theChars)  # 调用generate_image
                image_batch.append(self.process_(image))
        else:
            for i_ in ind_list:
                theChars = gen_rand(i_)
                code = [SPACE_INDEX if theChars == SPACE_TOKEN else encode_maps[c] for c in list(theChars)]
                label_batch.append(code)
                image = self.captcha.generate_image(theChars)  # 调用generate_image
                image_batch.append(self.process_(image))

        def get_input_lens(sequences):
            lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
            return sequences, lengths
        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        # batch_inputs,batch_seq_len = pad_input_sequences(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_len, batch_labels

    # 获取label
    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])
        return labels

    # 图片读取（用于一边生成图片一边训练的方法）
    def process_(self, image):
        im = np.array(image, dtype=np.uint8).astype(np.float32) / 255.
        im = im.swapaxes(0, 1)

        return im

    # 图片读取（用于已经生成好的图片）
    def process_image(self, image_name):
        im = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), 0).astype(np.float32)  # python 3 读图片
        im = im / 255.
        # cv2.imshow('{}'.format(image_name), im)

        im = cv2.resize(im, (image_width, image_height))

        im = im.swapaxes(0, 1)
        im = np.array(im)
        # print(im.shape)
        return im

    def input_index_generate_batch(self, index=None):
        if self.data_dir is None:
            # batch_inputs, batch_seq_len, batch_labels = self.generate_batch_img(FLAGS.batch_size)
            batch_inputs, batch_seq_len, batch_labels = self.generate_batch_img(ind_list=index)
            return batch_inputs, batch_seq_len, batch_labels

        if index:
            image_batch = [self.process_image(self.image_names[i]) for i in index]
            # image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            # get the whole data as input
            image_batch = [self.process_image(i) for i in self.image_names]
            # image_batch = self.image
            label_batch = self.labels

        def get_input_lens(sequences):
            lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
            return sequences, lengths
        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        # batch_inputs,batch_seq_len = pad_input_sequences(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)
        return batch_inputs, batch_seq_len, batch_labels


# 准确率计算方法2
def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=True):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq,please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if isPrint and i < maxPrintLen:
            print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))
        if origin_label == decoded_label:
            count += 1
    return count*1.0/len(original_seq)


# 准确率计算方法2
def accuracy_calculation2(name_seq, original_seq, decoded_seq, ignore_value=-1, isPrint=True):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq,please check again')
        return 0
    acc = []
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        decoded_seq_ = [decode_maps[k] for k in decoded_label]
        decoded_seq_ = ''.join(decoded_seq_)
        if isPrint and i < maxPrintLen:
            print('seq {0:4d}: {1} name:{2}'.format(i, decoded_seq_, name_seq[i]))
            # print('seq {0:4d}: origin: {1} decoded:{2} name:{3}'.format(i, origin_label, decoded_label, name_seq[i]))
        if len(decoded_label) == len(origin_label):
            equ = np.int_(np.equal(np.array(origin_label), np.array(decoded_label)))
            acc.append(np.sum(equ)/len(decoded_label))
        else:
            acc.append(0.0)
    return np.mean(acc)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape


def pad_input_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''
    Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.
        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''

    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths