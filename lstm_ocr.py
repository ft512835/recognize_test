# -*- coding:utf8 -*-

import os, sys
# import config
import numpy as np
import tensorflow as tf
import random
import cv2, time
import logging, datetime
from tensorflow.python.client import device_lib
from tensorflow.python.client import timeline
import utils
from tensorflow.python.training import moving_averages
from math import ceil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

FLAGS = utils.FLAGS
num_classes = utils.num_classes
num_features = utils.num_features

logger = logging.getLogger('Traing for ocr using LSTM+CTC')
logger.setLevel(logging.INFO)


# with tf.get_default_graph()._kernel_label_map({'CTCLoss':'WarpCTC'}):
# with tf.device('/gpu:1'):

# 无法收敛
class LSTM_CTC(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # e.g: log filter bank or MFCC features
            # Has size [batch_size, max_stepsize, num_features], but the
            # batch_size and max_stepsize can vary along each step
            # self.inputs = tf.placeholder(tf.float32,
            #                              [None, None, None, FLAGS.image_channel])
            self.inputs = tf.placeholder(tf.float32, [None, utils.image_width, num_features])
            # shape_ = tf.shape(self.inputs)
            # batch_s_, max_timesteps_ = shape_[0], shape_[1]
            #
            # self.inputs = tf.reshape(self.inputs, shape=[batch_s_, num_features, max_timesteps_])

            # Here we use sparse_placeholder that will generate a
            # SparseTensor required by ctc_loss op.
            self.labels = tf.sparse_placeholder(tf.int32)

            # 1d array of size [batch_size]
            self.seq_len = tf.placeholder(tf.int32, [None])
            self.keep_drop = tf.placeholder(tf.float32)


            # ====================================================================================
            # self.inputs = tf.transpose(self.inputs, [0, 2, 1])
            # Defining the cell
            # Can be:
            #   tf.nn.rnn_cell.RNNCell
            #   tf.nn.rnn_cell.GRUCell

            # # option 0
            cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.keep_drop)

            # # Stacking rnn cells
            stack = tf.contrib.rnn.MultiRNNCell([cell]*2, state_is_tuple=True)

            # # option 1
            # stack = tf.contrib.rnn.MultiRNNCell(
            #     [tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True) for _ in range(FLAGS.num_layers)],
            #     state_is_tuple=True)

            # The second output is the last state and we will no use that
            outputs, _ = tf.nn.dynamic_rnn(stack, self.inputs, self.seq_len, dtype=tf.float32)

            # ====================================================================================

            shape = tf.shape(self.inputs)
            batch_s, max_timesteps = shape[0], shape[1]

            # ====================================================================================

            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])

            # Truncated normal with mean 0 and stdev=0.1
            # Tip: Try another initialization
            # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden,
                                                 num_classes],
                                                stddev=0.1, dtype=tf.float32), name='W')

            # Zero initialization
            # Tip: Is tf.zeros_initializer the same?
            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))

            # Doing the affine projection
            logits = tf.matmul(outputs, W) + b

            # Reshaping back to the original shape
            logits = tf.reshape(logits, [batch_s, -1, num_classes])

            # Time major
            logits = tf.transpose(logits, (1, 0, 2))

            self.global_step = tf.Variable(0, trainable=False)

            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=self.seq_len)
            self.cost = tf.reduce_mean(self.loss)

            self.learning_rate = tf.maximum(0.000001, tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                            self.global_step,
                                                            FLAGS.decay_steps,
                                                            FLAGS.decay_rate, staircase=True))

            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
            #        momentum=FLAGS.momentum).minimize(self.cost,global_step=self.global_step)

            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum,
            #                                      use_nesterov=True).minimize(self.cost, global_step=self.global_step)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                  beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(self.loss, global_step=self.global_step)

            # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
            # (it's slower but you'll get better results)
            # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
            # Inaccuracy: label error rate
            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)
            # tf.summary.scalar('lerr',self.lerr)
            self.merged_summay = tf.summary.merge_all()

    def BiRNN(self, x):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # n_steps = tf.shape(x)[1]

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, utils.image_width, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return outputs


# 简单数据可行
class BiLSTM_LSTM_CTC(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # e.g: log filter bank or MFCC features
            # Has size [batch_size, max_stepsize, num_features], but the
            # batch_size and max_stepsize can vary along each step
            # self.inputs = tf.placeholder(tf.float32,
            #                              [None, None, None, FLAGS.image_channel])
            self.inputs = tf.placeholder(tf.float32, [None, utils.image_width, num_features])
            # shape_ = tf.shape(self.inputs)
            # batch_s_, max_timesteps_ = shape_[0], shape_[1]
            #
            # self.inputs = tf.reshape(self.inputs, shape=[batch_s_, num_features, max_timesteps_])

            # Here we use sparse_placeholder that will generate a
            # SparseTensor required by ctc_loss op.
            self.labels = tf.sparse_placeholder(tf.int32)

            # 1d array of size [batch_size]
            self.seq_len = tf.placeholder(tf.int32, [None])
            self.keep_drop = tf.placeholder(tf.float32)

            # ============================BiLSTM=========================================

            self.inputs_ = self.BiRNN(self.inputs)
            # self.inputs_ = tf.expand_dims(self.inputs_, -1)
            self.inputs_ = tf.transpose(self.inputs_, [1, 0, 2])
            # self.inputs_ = tf.tile(self.inputs_, [utils.image_width, 1])

            # ====================================================================================
            # self.inputs = tf.transpose(self.inputs, [0, 2, 1])
            # Defining the cell
            # Can be:
            #   tf.nn.rnn_cell.RNNCell
            #   tf.nn.rnn_cell.GRUCell

            # # option 0
            cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.keep_drop)

            cell1 = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            cell1 = tf.contrib.rnn.DropoutWrapper(cell=cell1, output_keep_prob=self.keep_drop)

            # # Stacking rnn cells
            stack = tf.contrib.rnn.MultiRNNCell([cell, cell1], state_is_tuple=True)

            # # option 1
            # stack = tf.contrib.rnn.MultiRNNCell(
            #     [tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True) for _ in range(FLAGS.num_layers)],
            #     state_is_tuple=True)

            # The second output is the last state and we will no use that
            outputs, _ = tf.nn.dynamic_rnn(stack, self.inputs_, self.seq_len, dtype=tf.float32)

            # ====================================================================================

            shape = tf.shape(self.inputs_)
            batch_s, max_timesteps = shape[0], shape[1]

            # ====================================================================================

            # # # option 2
            # x = tf.unstack(self.inputs, utils.image_width, axis=1)
            # lstm_fw_cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, initializer=tf.truncated_normal_initializer(stddev=0.01))
            # # Backward direction cell
            # lstm_bw_cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, initializer=tf.truncated_normal_initializer(stddev=0.01))
            #
            # # Get lstm cell output
            # outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
            #                                                             dtype=tf.float32)
            # outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            #     lstm_fw_cell, lstm_bw_cell, x,
            #     dtype=tf.float32)
            # ====================================================================================

            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])

            # Truncated normal with mean 0 and stdev=0.1
            # Tip: Try another initialization
            # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden,
                                                 num_classes],
                                                stddev=0.1, dtype=tf.float32), name='W')

            # Zero initialization
            # Tip: Is tf.zeros_initializer the same?
            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))

            # Doing the affine projection
            logits = tf.matmul(outputs, W) + b

            # Reshaping back to the original shape
            logits = tf.reshape(logits, [batch_s, -1, num_classes])

            # Time major
            logits = tf.transpose(logits, (1, 0, 2))

            self.global_step = tf.Variable(0, trainable=False)

            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=self.seq_len)
            self.cost = tf.reduce_mean(self.loss)

            self.learning_rate = tf.maximum(0.000001, tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                                                 self.global_step,
                                                                                 FLAGS.decay_steps,
                                                                                 FLAGS.decay_rate, staircase=True))

            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
            #        momentum=FLAGS.momentum).minimize(self.cost,global_step=self.global_step)

            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum,
            #                                      use_nesterov=True).minimize(self.cost, global_step=self.global_step)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                                                    beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(self.loss,
                                                                                                   global_step=self.global_step)

            # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
            # (it's slower but you'll get better results)
            # decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
            # Inaccuracy: label error rate
            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)
            # tf.summary.scalar('lerr',self.lerr)
            self.merged_summay = tf.summary.merge_all()

    def BiRNN(self, x):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # n_steps = tf.shape(x)[1]

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, utils.image_width, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                                    dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                              dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return outputs


# 无法收敛
class CNN_LSTM_CTC(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.inputs = tf.placeholder(tf.float32, [None, None, num_features])
            inputs = tf.expand_dims(self.inputs, -1)   # shape: [b, w, h, c]

            self.labels = tf.sparse_placeholder(tf.int32)
            self.seq_len = tf.placeholder(tf.int32, [None])
            self.keep_drop = tf.placeholder(tf.float32)
            # =============================CNN=========================================

            batch_s = tf.shape(inputs)[0]
            # conv = tf.reshape(inputs, shape=[batch_s, num_features, -1, 1])  # shape: [b, h, w, c]
            # conv = tf.reshape(inputs, shape=[batch_s, -1, num_features, 1])  # shape: [b, w, h, c]

            # 卷积层1
            w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 64]))
            b_conv1 = tf.Variable(tf.constant(0., shape=[64]))
            conv1 = tf.nn.conv2d(inputs, w_conv1, strides=[1, 1, 1, 1], padding='SAME')  # shape: [b, w, h, c]
            # conv1 = self._leaky_relu(tf.nn.bias_add(conv1, b_conv1), 0.01)
            conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1))

            # 卷积层2
            w_conv2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128]))
            b_conv2 = tf.Variable(tf.constant(0., shape=[128]))
            conv2 = tf.nn.conv2d(conv1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')  # shape: [b, w, h, c]
            conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2))
            # conv2 = self._leaky_relu(tf.nn.bias_add(conv2, b_conv2), 0.01)

            # 池化层1
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(self.seq_len, 2)

            # 卷积层3
            w_conv3 = tf.Variable(tf.truncated_normal([3, 3, 128, 128]))
            b_conv3 = tf.Variable(tf.constant(0., shape=[128]))
            conv3 = tf.nn.conv2d(conv2, w_conv3, strides=[1, 1, 1, 1], padding='SAME')  # shape: [b, w/2, h/2, c]
            conv3 = tf.nn.relu(tf.nn.bias_add(conv3, b_conv3))
            # conv3 = self._leaky_relu(tf.nn.bias_add(conv3, b_conv3), 0.01)

            # 卷积层4
            # w_conv4 = tf.Variable(tf.truncated_normal([3, 3, 128, 128]))
            # b_conv4 = tf.Variable(tf.constant(0., shape=[128]))
            # conv4 = tf.nn.conv2d(conv3, w_conv4, strides=[1, 1, 1, 1], padding='SAME')  # shape: [b, w, h, c]
            # # conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2))
            # conv4 = self._leaky_relu(tf.nn.bias_add(conv4, b_conv4), 0.01)

            # 池化层2
            conv4 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   # shape: [b, w/4, h/4, c]
            seq_len = tf.floor_div(seq_len, 2)
            # xx = tf.transpose(conv, (0, 2, 1, 3))  # (time/2, batc  h, features/2, channels==32)  # shape: [b, w, h, c]

            # n_f = tf.shape(conv4)[2]
            xx = tf.reshape(conv4, [batch_s, -1, 8*128])  # (batch, time/2, features/2 * 32) aka: [b, w, h*c]

            # ============================LSTM=========================================

            # Defining the cell
            # Can be:
            #   tf.nn.rnn_cell.RNNCell
            #   tf.nn.rnn_cell.GRUCell

            # option 1
            cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, initializer=tf.truncated_normal_initializer(stddev=0.01),
                                           state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.keep_drop)

            cell1 = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, initializer=tf.truncated_normal_initializer(stddev=0.01),
                                            state_is_tuple=True)
            cell1 = tf.contrib.rnn.DropoutWrapper(cell=cell1, output_keep_prob=self.keep_drop)
            # Stacking rnn cells
            stack = tf.contrib.rnn.MultiRNNCell([cell, cell1], state_is_tuple=True)

            # option 2
            # cell = tf.contrib.rnn_cell.GRUCell(FLAGS.num_hidden, state_is_tuple=True)
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)
            #
            # cell1 = tf.contrib.rnn_cell.GRUCell(FLAGS.num_hidden, state_is_tuple=True)
            # cell1 = tf.contrib.rnn.DropoutWrapper(cell=cell1, output_keep_prob=0.8)
            # # Stacking rnn cells
            # stack = tf.contrib.rnn.MultiRNNCell([cell, cell1], state_is_tuple=True)

            # option 3
            # stack = tf.contrib.rnn.MultiRNNCell(
            #     [tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True) for _ in range(FLAGS.num_layers)],
            #     state_is_tuple=True)

            outputs, _ = tf.nn.dynamic_rnn(stack, xx, seq_len, dtype=tf.float32)

            shape = tf.shape(xx)
            batch_s, max_timesteps = shape[0], shape[1]

            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])

            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden,
                                                 num_classes],
                                                stddev=0.1, dtype=tf.float32), name='W')

            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))

            logits = tf.matmul(outputs, W) + b
            logits = tf.reshape(logits, [batch_s, -1, num_classes])
            logits = tf.transpose(logits, (1, 0, 2))

            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=seq_len)
            self.cost = tf.reduce_mean(self.loss)
            # self.learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
            #                                                 self.global_step,
            #                                                 FLAGS.decay_steps,
            #                                                 FLAGS.decay_rate, staircase=True)
            self.learning_rate = tf.maximum(0.000001, tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                                                self.global_step,
                                                                                FLAGS.decay_steps,
                                                                                FLAGS.decay_rate, staircase=True))
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum,
            #                                      use_nesterov=True).minimize(self.cost, global_step=self.global_step)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                  beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(self.loss, global_step=self.global_step)

            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)

            self.merged_summay = tf.summary.merge_all()

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def BiRNN(self, x, weights, biases):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        n_steps = tf.shape(x)[1]
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, n_steps, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']


# 效果不错
class CNN_BiLSTM_LSTM_CTC(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.inputs = tf.placeholder(tf.float32, [None, None, num_features])
            inputs = tf.expand_dims(self.inputs, -1)   # shape: [b, w, h, c]

            self.labels = tf.sparse_placeholder(tf.int32)
            self.seq_len = tf.placeholder(tf.int32, [None])
            self.keep_drop = tf.placeholder(tf.float32)
            self.width = utils.image_width
            self.height = utils.image_height
            # self.keep_drop = tf.placeholder(tf.float32)
            self.rnn_keep_drop = tf.placeholder(tf.float32)
            # =============================CNN=========================================

            batch_s = tf.shape(inputs)[0]

            # 卷积层1
            with tf.variable_scope('unit-1'):
                w_conv = tf.get_variable(name='DW1',
                                         shape=[3, 3, 1, 128],
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())

                b_conv = tf.get_variable(name='bais1',
                                    shape=[128],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer())
                # w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 64]))
                # b_conv1 = tf.Variable(tf.constant(0., shape=[64]))
                conv = tf.nn.conv2d(inputs, w_conv, strides=[1, 1, 1, 1], padding='SAME')  # shape: [b, w, h, c]
                conv = self._leaky_relu(tf.nn.bias_add(conv, b_conv), 0.01)
                # conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1))


            # 池化层1
            conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')  # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(self.seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            # 卷积层2
            with tf.variable_scope('unit-2'):
                w_conv = tf.get_variable(name='DW2',
                                         shape=[3, 3, 128, 128],
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())

                b_conv = tf.get_variable(name='bais2',
                                         shape=[128],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer())

                conv = tf.nn.conv2d(conv, w_conv, strides=[1, 1, 1, 1], padding='SAME')  # shape: [b, w, h, c]
                conv = self._leaky_relu(tf.nn.bias_add(conv, b_conv), 0.01)

            # 池化层2
            conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')  # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            # 卷积层3
            with tf.variable_scope('unit-3'):
                w_conv = tf.get_variable(name='DW3',
                                         shape=[3, 3, 128, 256],
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())

                b_conv = tf.get_variable(name='bais3',
                                         shape=[256],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer())

                conv = tf.nn.conv2d(conv, w_conv, strides=[1, 1, 1, 1], padding='SAME')  # shape: [b, w/2, h/2, c]
                conv = self._leaky_relu(tf.nn.bias_add(conv, b_conv), 0.01)

            # 池化层3
            conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')   # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)


            # 卷积层4
            with tf.variable_scope('unit-4'):
                w_conv = tf.get_variable(name='DW4',
                                         shape=[3, 3, 256, 512],
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.xavier_initializer())

                b_conv = tf.get_variable(name='bais4',
                                         shape=[512],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer())

                conv = tf.nn.conv2d(conv, w_conv, strides=[1, 1, 1, 1], padding='SAME')  # shape: [b, w/2, h/2, c]
                conv = self._leaky_relu(tf.nn.bias_add(conv, b_conv), 0.01)

            # 池化层4
            conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')  # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            # xx = tf.transpose(conv, (0, 2, 1, 3))  # shape: [b, w, h, c]

            # n_f = tf.shape(conv4)[2]
            xx = tf.reshape(conv, [batch_s, -1, self.height*512])  # (batch, time/2, features/2 * 32) aka: [b, w, h*c]

            # ============================BiLSTM=========================================

            xx = self.BiRNN(xx)
            xx = tf.transpose(xx, [1, 0, 2])

            # ============================LSTM=========================================

            # Defining the cell
            # Can be:
            #   tf.nn.rnn_cell.RNNCell
            #   tf.nn.rnn_cell.GRUCell
            #
            # cell = tf.contrib.rnn_cell.GRUCell(FLAGS.num_hidden, state_is_tuple=True)
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)

            cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, initializer=tf.truncated_normal_initializer(stddev=0.01),
                                           state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.keep_drop)

            # Stacking rnn cells
            stack = tf.contrib.rnn.MultiRNNCell([cell]*1, state_is_tuple=True)

            # option 2
            # stack = tf.contrib.rnn.MultiRNNCell(
            #     [tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True) for _ in range(FLAGS.num_layers)],
            #     state_is_tuple=True)

            outputs, _ = tf.nn.dynamic_rnn(stack, xx, seq_len, dtype=tf.float32)

            shape = tf.shape(xx)
            batch_s, max_timesteps = shape[0], shape[1]

            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])

            # ============================FC=========================================

            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden,
                                                 num_classes],
                                                stddev=0.1, dtype=tf.float32), name='W')

            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))

            logits = tf.matmul(outputs, W) + b

            logits = tf.reshape(logits, [batch_s, -1, num_classes])
            logits = tf.transpose(logits, (1, 0, 2))  # time major

            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=seq_len)
            self.cost = tf.reduce_mean(self.loss)

            self.learning_rate = tf.maximum(0.000001, tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                                                self.global_step,
                                                                                FLAGS.decay_steps,
                                                                                FLAGS.decay_rate, staircase=True))
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum,
            #                                      use_nesterov=True).minimize(self.cost, global_step=self.global_step)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                  beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(self.loss, global_step=self.global_step)

            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)

            self.merged_summay = tf.summary.merge_all()

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def BiRNN(self, x):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # n_steps = tf.shape(x)[1]
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        # x = tf.unstack(x, n_steps, 1)

        x = tf.unstack(x, self.width, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return outputs


class res_CNN_BiLSTM_LSTM_CTC(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.inputs = tf.placeholder(tf.float32, [None, None, num_features])
            inputs = tf.expand_dims(self.inputs, -1)   # shape: [b, w, h, c]

            self.labels = tf.sparse_placeholder(tf.int32)
            self.seq_len = tf.placeholder(tf.int32, [None])
            self.keep_drop = tf.placeholder(tf.float32)
            self.width = utils.image_width
            self.height = utils.image_height

            # =============================CNN=========================================

            batch_s = tf.shape(inputs)[0]

            conv = self.Conv_(inputs, [1, 256], 'unit-0')  # 卷积层0

            # 池化层0
            conv_ = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME')  # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(self.seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            conv = self.Conv_(conv_, [256, 256], 'unit-1')  # 卷积层1
            conv0_1 = conv + conv_  # 0 1 相加

            # 池化层1
            conv_ = tf.nn.max_pool(conv0_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME')  # shape: [b, w/4, h/4, c]
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            conv = self.Conv_(conv_, [256, 256], 'unit-2')  # 卷积层2
            conv1_2 = conv + conv_  # 1 2 相加

            # 池化层2
            conv_ = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')  # shape: [b, w/4, h/4, c]
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            conv = self.Conv_(conv_, [256, 256], 'unit-3')  # 卷积层3
            conv2_3 = conv + conv_  # 2 3 相加

            # 池化层3
            conv_ = tf.nn.max_pool(conv2_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME')  # shape: [b, w/4, h/4, c]
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            conv = self.Conv_(conv_, [256, 512], 'unit-4')  # 卷积层4

            # 池化层4
            # conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            #                        padding='SAME')  # shape: [b, w/2, h/2, c]
            # seq_len = tf.floor_div(seq_len, 2)
            # self.width = np.ceil(self.width / 2)
            # self.height = np.ceil(self.height / 2)

            # # 卷积层5 6
            # with tf.variable_scope('unit-5'):
            #     w_conv = tf.get_variable(name='DW3',
            #                              shape=[3, 3, 256, 256],
            #                              dtype=tf.float32,
            #                              initializer=tf.contrib.layers.xavier_initializer())
            #
            #     b_conv = tf.get_variable(name='bais3',
            #                              shape=[256],
            #                              dtype=tf.float32,
            #                              initializer=tf.constant_initializer())
            #
            #     conv = tf.nn.conv2d(conv_, w_conv, strides=[1, 1, 1, 1], padding='SAME')  # shape: [b, w/2, h/2, c]
            #     conv = self._leaky_relu(tf.nn.bias_add(conv, b_conv), 0.01)
            #
            #     conv = conv + conv_  # 4 5 相加
            #
            # # 卷积层5 6
            # with tf.variable_scope('unit-6'):
            #     w_conv = tf.get_variable(name='DW3-1',
            #                              shape=[3, 3, 256, 256],
            #                              dtype=tf.float32,
            #                              initializer=tf.contrib.layers.xavier_initializer())
            #
            #     b_conv = tf.get_variable(name='bais3-1',
            #                              shape=[256],
            #                              dtype=tf.float32,
            #                              initializer=tf.constant_initializer())
            #
            #     conv = tf.nn.conv2d(conv, w_conv, strides=[1, 1, 1, 1], padding='SAME')  # shape: [b, w/2, h/2, c]
            #     conv = self._leaky_relu(tf.nn.bias_add(conv, b_conv), 0.01)
            #
            #
            # # 池化层3
            # conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            #                        padding='SAME')   # shape: [b, w/2, h/2, c]
            # seq_len = tf.floor_div(seq_len, 2)
            # self.width = np.ceil(self.width / 2)
            # self.height = np.ceil(self.height / 2)


            # # 卷积层7 8
            # with tf.variable_scope('unit-4'):
            #     w_conv = tf.get_variable(name='DW4',
            #                              shape=[3, 3, 512, 512],
            #                              dtype=tf.float32,
            #                              initializer=tf.contrib.layers.xavier_initializer())
            #
            #     b_conv = tf.get_variable(name='bais4',
            #                              shape=[512],
            #                              dtype=tf.float32,
            #                              initializer=tf.constant_initializer())
            #
            #     conv = tf.nn.conv2d(conv_, w_conv, strides=[1, 1, 1, 1], padding='SAME')  # shape: [b, w/2, h/2, c]
            #     conv = self._leaky_relu(tf.nn.bias_add(conv, b_conv), 0.01)
            #
            #     conv = conv + conv_  # resnet_based
            #
            #     w_conv = tf.get_variable(name='DW4-1',
            #                          shape=[3, 3, 512, 512],
            #                          dtype=tf.float32,
            #                          initializer=tf.contrib.layers.xavier_initializer())
            #
            #     b_conv = tf.get_variable(name='bais4-1',
            #                          shape=[512],
            #                          dtype=tf.float32,
            #                          initializer=tf.constant_initializer())
            #
            #     conv = tf.nn.conv2d(conv, w_conv, strides=[1, 1, 1, 1], padding='SAME')  # shape: [b, w/2, h/2, c]
            #     conv = self._leaky_relu(tf.nn.bias_add(conv, b_conv), 0.01)
            #
            #
            # # 池化层4
            # conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            #                        padding='SAME')  # shape: [b, w/2, h/2, c]
            # seq_len = tf.floor_div(seq_len, 2)
            # self.width = np.ceil(self.width / 2)
            # self.height = np.ceil(self.height / 2)

            # xx = tf.transpose(conv, (0, 2, 1, 3))  # (time/2, batc  h, features/2, channels==32)  # shape: [b, w, h, c]

            # n_f = tf.shape(conv4)[2]
            xx = tf.reshape(conv, [batch_s, -1, self.height*512])  # (batch, time/2, features/2 * 32) aka: [b, w, h*c]

            # ============================BiLSTM=========================================

            xx = self.BiRNN(xx)
            xx = tf.transpose(xx, [1, 0, 2])

            # ============================LSTM=========================================

            # Defining the cell
            # Can be:
            #   tf.nn.rnn_cell.RNNCell
            #   tf.nn.rnn_cell.GRUCell
            #
            # cell = tf.contrib.rnn_cell.GRUCell(FLAGS.num_hidden, state_is_tuple=True)
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)

            cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, initializer=tf.truncated_normal_initializer(stddev=0.01),
                                           state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.keep_drop)

            # Stacking rnn cells
            stack = tf.contrib.rnn.MultiRNNCell([cell]*1, state_is_tuple=True)

            # option 2
            # stack = tf.contrib.rnn.MultiRNNCell(
            #     [tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True) for _ in range(FLAGS.num_layers)],
            #     state_is_tuple=True)

            outputs, _ = tf.nn.dynamic_rnn(stack, xx, seq_len, dtype=tf.float32)

            shape = tf.shape(xx)
            batch_s, max_timesteps = shape[0], shape[1]

            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])

            # ============================FC=========================================

            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden,
                                                 num_classes],
                                                stddev=0.1, dtype=tf.float32), name='W')

            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))

            logits = tf.matmul(outputs, W) + b

            logits = tf.reshape(logits, [batch_s, -1, num_classes])
            logits = tf.transpose(logits, (1, 0, 2))  # time major

            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=seq_len)
            self.cost = tf.reduce_mean(self.loss)

            self.learning_rate = tf.maximum(0.000001, tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                                                self.global_step,
                                                                                FLAGS.decay_steps,
                                                                                FLAGS.decay_rate, staircase=True))
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum,
            #                                      use_nesterov=True).minimize(self.cost, global_step=self.global_step)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                  beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(self.loss, global_step=self.global_step)

            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)

            self.merged_summay = tf.summary.merge_all()

    def Conv_(self, conv_, c, name, p=None):
        if p is None:
            p = [3, 1]

        with tf.variable_scope(name):
            w_conv = tf.get_variable(name='DW2',
                                     shape=[p[0], p[0], c[0], c[1]],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            b_conv = tf.get_variable(name='bais2',
                                     shape=[c[1]],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer())
            # w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 64]))
            # b_conv1 = tf.Variable(tf.constant(0., shape=[64]))
            conv = tf.nn.conv2d(conv_, w_conv, strides=[1, p[1], p[1], 1], padding='SAME')  # shape: [b, w, h, c]
            conv = self._leaky_relu(tf.nn.bias_add(conv, b_conv), 0.01)
            # conv = tf.nn.relu(tf.nn.bias_add(conv, b_conv))

            return conv

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def BiRNN(self, x):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # n_steps = tf.shape(x)[1]
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        # x = tf.unstack(x, n_steps, 1)

        x = tf.unstack(x, self.width, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return outputs


class CNN_BiLSTM_LSTM_CTC_2(object):
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.inputs = tf.placeholder(tf.float32, [None, None, num_features])
            inputs = tf.expand_dims(self.inputs, -1)   # shape: [b, w, h, c]

            self.labels = tf.sparse_placeholder(tf.int32)
            self.seq_len = tf.placeholder(tf.int32, [None])
            self.keep_drop = tf.placeholder(tf.float32)
            self.width = utils.image_width
            self.height = utils.image_height

            # =============================CNN=========================================

            batch_s = tf.shape(inputs)[0]

            conv = self.Conv_(inputs, [1, 128], 'unit-1')  # 卷积层1

            # 池化层1
            conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')  # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(self.seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            conv = self.Conv_(conv, [128, 128], 'unit-2')  # 卷积层2

            # 池化层2
            conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')  # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            conv = self.Conv_(conv, [128, 256], 'unit-3')  # 卷积层3

            # 池化层3
            conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')   # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            conv = self.Conv_(conv, [256, 512], 'unit-4')  # 卷积层4

            conv0 = self.Conv_(inputs, [1, 512], 'unit-0')  # 卷积层0
            conv0 = tf.nn.max_pool(conv0, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1],
                                   padding='SAME')  # shape: [b, w/2, h/2, c]
            # conv0 = tf.nn.avg_pool(conv0, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1],
            #                        padding='SAME')

            # conv = conv + conv0
            conv = tf.concat([conv, conv0], -1)
            # self.width = 2 * self.width

            # 池化层4
            conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')  # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            # xx = tf.transpose(conv, (0, 2, 1, 3))  # shape: [b, w, h, c]

            # n_f = tf.shape(conv4)[2]
            xx = tf.reshape(conv, [batch_s, -1, self.height*512*2])  # (batch, time/2, features/2 * 32) aka: [b, w, h*c]

            # ============================BiLSTM=========================================

            xx = self.BiRNN(xx)
            xx = tf.transpose(xx, [1, 0, 2])

            # ============================LSTM=========================================

            # Defining the cell
            # Can be:
            #   tf.nn.rnn_cell.RNNCell
            #   tf.nn.rnn_cell.GRUCell
            #
            # cell = tf.contrib.rnn_cell.GRUCell(FLAGS.num_hidden, state_is_tuple=True)
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)

            cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, initializer=tf.truncated_normal_initializer(stddev=0.01),
                                           state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.keep_drop)

            # Stacking rnn cells
            stack = tf.contrib.rnn.MultiRNNCell([cell]*1, state_is_tuple=True)

            # option 2
            # stack = tf.contrib.rnn.MultiRNNCell(
            #     [tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True) for _ in range(FLAGS.num_layers)],
            #     state_is_tuple=True)

            outputs, _ = tf.nn.dynamic_rnn(stack, xx, seq_len, dtype=tf.float32)

            shape = tf.shape(xx)
            batch_s, max_timesteps = shape[0], shape[1]

            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])

            # ============================FC=========================================

            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden,
                                                 num_classes],
                                                stddev=0.1, dtype=tf.float32), name='W')

            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))

            logits = tf.matmul(outputs, W) + b

            logits = tf.reshape(logits, [batch_s, -1, num_classes])
            logits = tf.transpose(logits, (1, 0, 2))  # time major

            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=seq_len)
            self.cost = tf.reduce_mean(self.loss)

            self.learning_rate = tf.maximum(0.000001, tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                                                self.global_step,
                                                                                FLAGS.decay_steps,
                                                                                FLAGS.decay_rate, staircase=True))
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum,
            #                                      use_nesterov=True).minimize(self.cost, global_step=self.global_step)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                  beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(self.loss, global_step=self.global_step)

            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)

            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)

            self.merged_summay = tf.summary.merge_all()

    def Conv_(self, conv_, c, name, p=None):
        if p is None:
            p = [3, 1]

        with tf.variable_scope(name):
            w_conv = tf.get_variable(name='DW2',
                                     shape=[p[0], p[0], c[0], c[1]],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            b_conv = tf.get_variable(name='bais2',
                                     shape=[c[1]],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer())
            # w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 64]))
            # b_conv1 = tf.Variable(tf.constant(0., shape=[64]))
            conv = tf.nn.conv2d(conv_, w_conv, strides=[1, p[1], p[1], 1], padding='SAME')  # shape: [b, w, h, c]
            conv = self._leaky_relu(tf.nn.bias_add(conv, b_conv), 0.01)
            # conv = tf.nn.relu(tf.nn.bias_add(conv, b_conv))

            return conv

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def BiRNN(self, x):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # n_steps = tf.shape(x)[1]
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        # x = tf.unstack(x, n_steps, 1)

        x = tf.unstack(x, self.width, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return outputs


class den_CNN_BiLSTM_LSTM_CTC(object):
    def __init__(self, mode='train'):
        self.mode = mode
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.inputs = tf.placeholder(tf.float32, [None, None, num_features])
            inputs = tf.expand_dims(self.inputs, -1)   # shape: [b, w, h, c]

            self.labels = tf.sparse_placeholder(tf.int32)
            self.seq_len = tf.placeholder(tf.int32, [None])
            self.keep_drop = tf.placeholder(tf.float32)
            self.width = utils.image_width
            self.height = utils.image_height

            # =============================CNN=========================================

            batch_s = tf.shape(inputs)[0]
            conv0 = self.Conv_(inputs, [1, 32], 'unit-0', p=[1, 1])  # 卷积层0-0
            # 池化层0
            conv_ = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')  # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(self.seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            conv0 = self.Conv_(conv_, [32, 32], 'unit-1')  # 卷积层0-1

            conv1 = self.Conv_(conv0, [32, 32], 'unit-2')  # 卷积层0-2
            conv01 = tf.concat([conv1, conv0], -1)  # 0 1 相加

            conv2 = self.Conv_(conv01, [64, 32], 'unit-3')  # 卷积层0-3
            conv012 = tf.concat([conv2, conv1, conv0], -1)  # 0 1 2 相加

            conv3 = self.Conv_(conv012, [96, 64], 'unit-4')  # 卷积层0-4
            conv0123 = tf.concat([conv3, conv2, conv1, conv0], -1)  # 0 1 2 3 相加

            # 池化层1
            conv_ = tf.nn.max_pool(conv0123, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME')  # shape: [b, w/2, h/2, c]
            # conv_ = tf.nn.dropout(conv_, keep_prob=self.keep_drop)
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)
            conv_ = self.Conv_(conv_, [160, 64], 'unit1-0', p=[1, 1])  # 卷积层1-0

            conv0 = self.Conv_(conv_, [64, 64], 'unit1-1')  # 卷积层1-1

            conv1 = self.Conv_(conv0, [64, 64], 'unit1-2')  # 卷积层1-2
            conv01 = tf.concat([conv1, conv0], -1)  # 0 1 相加

            conv2 = self.Conv_(conv01, [128, 64], 'unit1-3')  # 卷积层1-3
            conv012 = tf.concat([conv2, conv1, conv0], -1)  # 0 1 2 相加

            conv3 = self.Conv_(conv012, [192, 128], 'unit1-4')  # 卷积层1-4
            conv0123 = tf.concat([conv3, conv2, conv1, conv0], -1)  # 0 1 2 3 相加

            # 池化层2
            conv = tf.nn.max_pool(conv0123, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME')  # shape: [b, w/4, h/4, c]

            # conv = tf.nn.dropout(conv, keep_prob=self.keep_drop)
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            # conv = self.Conv_(conv_, [320, 128], 'unit2-0', p=[1, 1])  # 卷积层2-0

            # conv0 = self.Conv_(conv_, [128, 128], 'unit2-1')  # 卷积层2-1
            #
            # conv1 = self.Conv_(conv0, [128, 128], 'unit2-2')  # 卷积层2-2
            # conv01 = tf.concat([conv1, conv0], -1)  # 0 1 相加
            #
            # conv2 = self.Conv_(conv01, [256, 128], 'unit2-3')  # 卷积层2-3
            # conv012 = tf.concat([conv2, conv1, conv0], -1)  # 0 1 2 相加
            #
            # conv3 = self.Conv_(conv012, [384, 256], 'unit2-4')  # 卷积层2-4
            # conv0123 = tf.concat([conv3, conv2, conv1, conv0], -1)  # 0 1 2 3 相加
            #
            # # 池化层2
            # conv_ = tf.nn.max_pool(conv0123, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            #                        padding='SAME')  # shape: [b, w/4, h/4, c]
            # seq_len = tf.floor_div(seq_len, 2)
            # self.width = np.ceil(self.width / 2)
            # self.height = np.ceil(self.height / 2)
            #
            # conv = self.Conv_(conv_, [640, 512], 'unit3-0', p=[1, 1])  # 卷积层3-0

            # # 池化层3
            # conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            #                            padding='SAME')  # shape: [b, w/4, h/4, c]
            # seq_len = tf.floor_div(seq_len, 2)
            # self.width = np.ceil(self.width / 2)
            # self.height = np.ceil(self.height / 2)

            # 待测
            # conv = self.Conv_(conv_, [320, 512], 'unit-4')  # 卷积层4
            #
            # # 池化层3
            # conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            #                            padding='SAME')  # shape: [b, w/4, h/4, c]
            # seq_len = tf.floor_div(seq_len, 2)
            # self.width = np.ceil(self.width / 2)
            # self.height = np.ceil(self.height / 2)

            # xx = tf.transpose(conv, (0, 2, 1, 3))  # (time/2, batc  h, features/2, channels==32)  # shape: [b, w, h, c]

            # n_f = tf.shape(conv4)[2]
            xx = tf.reshape(conv, [batch_s, -1, self.height*320])  # (batch, time/2, features/2 * 32) aka: [b, w, h*c]

            # ============================BiLSTM=========================================

            xx = self.BiRNN(xx)
            xx = tf.transpose(xx, [1, 0, 2])

            # ============================LSTM=========================================

            # Defining the cell
            # Can be:
            #   tf.nn.rnn_cell.RNNCell
            #   tf.nn.rnn_cell.GRUCell
            #
            # cell = tf.contrib.rnn_cell.GRUCell(FLAGS.num_hidden, state_is_tuple=True)
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)

            cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, initializer=tf.truncated_normal_initializer(stddev=0.01),
                                           state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.keep_drop)

            # Stacking rnn cells
            stack = tf.contrib.rnn.MultiRNNCell([cell]*1, state_is_tuple=True)

            # option 2
            # stack = tf.contrib.rnn.MultiRNNCell(
            #     [tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True) for _ in range(FLAGS.num_layers)],
            #     state_is_tuple=True)

            outputs, _ = tf.nn.dynamic_rnn(stack, xx, seq_len, dtype=tf.float32)
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])

            # ============================FC=========================================

            shape = tf.shape(xx)
            batch_s, max_timesteps = shape[0], shape[1]

            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden,
                                                 num_classes],
                                                stddev=0.1, dtype=tf.float32), name='W')
            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))

            logits = tf.matmul(outputs, W) + b
            logits = tf.reshape(logits, [batch_s, -1, num_classes])
            logits = tf.transpose(logits, (1, 0, 2))  # time major

            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=seq_len)
            self.cost = tf.reduce_mean(self.loss)
            self.learning_rate = tf.maximum(0.000001, tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                                                self.global_step,
                                                                                FLAGS.decay_steps,
                                                                                FLAGS.decay_rate, staircase=True))
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum,
            #                                      use_nesterov=True).minimize(self.cost, global_step=self.global_step)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                  beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(self.loss, global_step=self.global_step)

            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)
            self.merged_summay = tf.summary.merge_all()

    def Conv_(self, conv_, c, name, p=None):
        if p is None:
            p = [3, 1]

        with tf.variable_scope(name):
            w_conv = tf.get_variable(name='DW2',
                                     shape=[p[0], p[0], c[0], c[1]],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            b_conv = tf.get_variable(name='bais2',
                                     shape=[c[1]],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer())
            # w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 64]))
            # b_conv1 = tf.Variable(tf.constant(0., shape=[64]))
            conv = tf.nn.conv2d(conv_, w_conv, strides=[1, p[1], p[1], 1], padding='SAME')  # shape: [b, w, h, c]
            conv = self._leaky_relu(tf.nn.bias_add(conv, b_conv), 0.01)
            # conv = tf.nn.relu(tf.nn.bias_add(conv, b_conv))

            return conv

    def batch_norm(self, x):
        if self.mode == 'train':
            # from tensorflow.contrib.layers import batch_norm
            return tf.contrib.layers.batch_norm(x, is_training=True)
        else:
            return tf.contrib.layers.batch_norm(x, is_training=False)

    def _batch_norm(self, x, name):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

            # tf.summary.histogram(mean.op.name, mean)
            # tf.summary.histogram(variance.op.name, variance)

            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())

            return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def BiRNN(self, x):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # n_steps = tf.shape(x)[1]
        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        # x = tf.unstack(x, n_steps, 1)

        x = tf.unstack(x, self.width, 1)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                   dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return outputs


den_block_1 = [32, 32, 64, 96, 160]
den_block_2 = [128, 128, 256, 384, 512]
class den_CNN_BN_BiLSTM_LSTM_CTC(object):
    def __init__(self, mode='train'):
        self.mode = mode
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.inputs = tf.placeholder(tf.float32, [None, None, num_features])
            inputs = tf.expand_dims(self.inputs, -1)   # shape: [b, w, h, c]

            self.labels = tf.sparse_placeholder(tf.int32)
            self.seq_len = tf.placeholder(tf.int32, [None])
            self.keep_drop = tf.placeholder(tf.float32)
            self.rnn_keep_drop = tf.placeholder(tf.float32)
            self.width = utils.image_width
            self.height = utils.image_height

            # ==============================================================================
            # CNN
            # ==============================================================================

            batch_s = tf.shape(inputs)[0]
            # ============================= conv + max pool 0 ===========================
            conv0 = self.Conv_(inputs, [1, 32], 'unit-0', p=[1, 1], bn=True)  # 卷积层0-0

            conv_ = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                   padding='SAME')  # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(self.seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            # ============================= dense block 0 ===========================

            # with tf.variable_scope('block-1'):
            conv0_0 = self.Conv_(conv_, [32, 32], 'unit-1')  # 卷积层0-1
            conv0_1 = self.Conv_(conv0_0, [32, 32], 'unit-2')  # 卷积层0-2
            conv0_01 = tf.concat([conv0_1, conv0_0], -1)  # 0 1 相加
            # conv01 = self._batch_norm(conv01, name='bn-1')  # BN
            conv0_2 = self.Conv_(conv0_01, [64, 32], 'unit-3')  # 卷积层0-3
            conv0_012 = tf.concat([conv0_2, conv0_1, conv0_0], -1)  # 0 1 2 相加
            # conv012 = self._batch_norm(conv012, name='bn-2')  # BN
            conv0_3 = self.Conv_(conv0_012, [96, 64], 'unit-4')  # 卷积层0-4
            conv0_0123 = tf.concat([conv0_3, conv0_2, conv0_1, conv0_0], -1)  # 0 1 2 3 相加
            # conv0_0123 = self.batch_norm(conv0_0123)  # BN

            conv_ = tf.nn.dropout(conv0_0123, keep_prob=self.keep_drop)

            # ============================= conv + max pool 1 ===========================

            conv_ = self.Conv_(conv_, [160, 128], 'unit1-0', p=[1, 1], bn=True)  # 卷积层1-0

            conv_ = tf.nn.max_pool(conv_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME')  # shape: [b, w/2, h/2, c]
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)

            # ============================= dense block 1 ===========================

            conv1_0 = self.Conv_(conv_, [128, 128], 'unit1-1')  # 卷积层1-1
            conv1_1 = self.Conv_(conv1_0, [128, 128], 'unit1-2')  # 卷积层1-2
            conv1_01 = tf.concat([conv1_1, conv1_0], -1)  # 0 1 相加
            # conv01 = self._batch_norm(conv01, name='bn1-1')  # BN
            conv1_2 = self.Conv_(conv1_01, [256, 128], 'unit1-3')  # 卷积层1-3
            conv1_012 = tf.concat([conv1_2, conv1_1, conv1_0], -1)  # 0 1 2 相加
            # conv012 = self._batch_norm(conv012, name='bn1-2')  # BN
            conv1_3 = self.Conv_(conv1_012, [384, 128], 'unit1-4')  # 卷积层1-4
            conv1_0123 = tf.concat([conv1_3, conv1_2, conv1_1, conv1_0], -1)  # 0 1 2 3 相加
            # conv1_0123 = self.batch_norm(conv1_0123)  # BN

            conv_ = tf.nn.dropout(conv1_0123, keep_prob=self.keep_drop)

            # ============================= max pool 2 ===========================

            conv = tf.nn.max_pool(conv_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME')  # shape: [b, w/4, h/4, c]
            seq_len = tf.floor_div(seq_len, 2)
            self.width = np.ceil(self.width / 2)
            self.height = np.ceil(self.height / 2)


            # 待测
            # conv = self.Conv_(conv_, [320, 512], 'unit-4')  # 卷积层4
            #
            # # 池化层3
            # conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            #                            padding='SAME')  # shape: [b, w/4, h/4, c]
            # seq_len = tf.floor_div(seq_len, 2)
            # self.width = np.ceil(self.width / 2)
            # self.height = np.ceil(self.height / 2)

            # n_f = tf.shape(conv4)[2]
            xx = tf.reshape(conv, [batch_s, -1, self.height*512])  # (batch, time/2, features/2 * 32) aka: [b, w, h*c]

            # ==============================================================================
            # BiLSTM
            # ==============================================================================

            xx = self.BiRNN(xx)
            xx = tf.transpose(xx, [1, 0, 2])

            # ==============================================================================
            # LSTM
            # ==============================================================================

            # Defining the cell
            # Can be:
            #   tf.nn.rnn_cell.RNNCell
            #   tf.nn.rnn_cell.GRUCell
            #
            # cell = tf.contrib.rnn_cell.GRUCell(FLAGS.num_hidden, state_is_tuple=True)
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.8)

            cell = tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, initializer=tf.truncated_normal_initializer(stddev=0.01),
                                           state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.rnn_keep_drop)  # dropout
            stack = tf.contrib.rnn.MultiRNNCell([cell]*1, state_is_tuple=True)

            # option 2
            # stack = tf.contrib.rnn.MultiRNNCell(
            #     [tf.contrib.rnn.LSTMCell(FLAGS.num_hidden, state_is_tuple=True) for _ in range(FLAGS.num_layers)],
            #     state_is_tuple=True)

            outputs, _ = tf.nn.dynamic_rnn(stack, xx, seq_len, dtype=tf.float32)
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])

            # ==============================================================================
            # FC
            # ==============================================================================

            shape = tf.shape(xx)
            batch_s, max_timesteps = shape[0], shape[1]

            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden,
                                                 num_classes],
                                                stddev=0.1, dtype=tf.float32), name='W')
            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))

            logits = tf.matmul(outputs, W) + b
            logits = tf.reshape(logits, [batch_s, -1, num_classes])
            logits = tf.transpose(logits, (1, 0, 2))  # time major

            # ==============================================================================
            # Optimizer and Loss
            # ==============================================================================

            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=seq_len)
            self.cost = tf.reduce_mean(self.loss)
            self.learning_rate = tf.maximum(0.000001, tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                                                self.global_step,
                                                                                FLAGS.decay_steps,
                                                                                FLAGS.decay_rate, staircase=True))
            # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum,
            #                                      use_nesterov=True).minimize(self.cost, global_step=self.global_step)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate,
                    beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(self.loss, global_step=self.global_step)

            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)
            self.merged_summay = tf.summary.merge_all()

    def Conv_(self, conv_, c, name, p=None, bn=False):
        if p is None:
            p = [3, 1]

        with tf.variable_scope(name):
            w_conv = tf.get_variable(name='DW',
                                     shape=[p[0], p[0], c[0], c[1]],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer())

            b_conv = tf.get_variable(name='bais',
                                     shape=[c[1]],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer())
            # w_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 64]))
            # b_conv1 = tf.Variable(tf.constant(0., shape=[64]))
            conv = tf.nn.conv2d(conv_, w_conv, strides=[1, p[1], p[1], 1], padding='SAME')  # shape: [b, w, h, c]
            conv = tf.nn.bias_add(conv, b_conv)
            if bn:
                conv = self.batch_norm(conv)
            conv = self._leaky_relu(conv, 0.1)
            # conv = self._Swish(conv)
            # conv = tf.nn.relu(tf.nn.bias_add(conv, b_conv))

            return conv

    def batch_norm(self, x):
        if self.mode == 'train':
            # from tensorflow.contrib.layers import batch_norm
            return tf.contrib.layers.batch_norm(x, is_training=True)
        else:
            return tf.contrib.layers.batch_norm(x, is_training=False)

    def _batch_norm(self, x, name):
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable(
                'beta', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable(
                'gamma', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            else:
                mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False)
                variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False)

            # tf.summary.histogram(mean.op.name, mean)
            # tf.summary.histogram(variance.op.name, variance)

            # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())

            return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _Swish(self, x):
        return tf.multiply(x, tf.nn.sigmoid(x), name='Swish')

    def _Swish_relu(self, x):
        return tf.where(tf.less(x, 0.0), x*tf.nn.sigmoid(x), x, name='swish_relu')

    def BiRNN(self, x):
        # input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # width is the n_steps
        x = tf.unstack(x, self.width, 1)

        # Forward direction cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.num_hidden, forget_bias=1.0)

        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                         dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return outputs


