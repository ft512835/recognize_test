#-*- coding:utf8 -*-

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
from lstm_ocr import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

FLAGS = utils.FLAGS
num_classes = utils.num_classes
num_features = utils.num_features

logger = logging.getLogger('Traing for ocr using CNN+BiLSTM+LSTM+CTC')
logger.setLevel(logging.INFO)


def train(train_dir=None, val_dir=None):
    # g = Graph()
    # g = BiLSTM_LSTM()
    # g = CNN_BiLSTM_LSTM_CTC()
    # g = res_CNN_BiLSTM_LSTM_CTC()
    # g = den_CNN_BiLSTM_LSTM_CTC()
    g = den_CNN_BN_BiLSTM_LSTM_CTC()

    # g = CNN_LSTM_CTC()
    # with g.graph.as_default():
    print('loading train data, please wait---------------------', end=' ')
    # train_feeder = utils.DataIterator(data_dir=train_dir)
    train_feeder = utils.DataIterator()  # 一边生成数据 一边训练
    print('get image: ', train_feeder.size)

    print('loading validation data, please wait---------------------', end=' ')
    val_feeder = utils.DataIterator(data_dir=val_dir)
    print('get image: ', val_feeder.size)

    num_train_samples = train_feeder.size  # 256000
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size)  # example: 256000/256

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)
    with tf.Session(graph=g.graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        g.graph.finalize()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # the global_step will restore as well
                saver.restore(sess, ckpt)
                print('restore from the checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')
        # the cuda trace
        # run_metadata = tf.RunMetadata()
        # trace_file = open('timeline.ctf.json','w')

        val_inputs, val_seq_len, val_labels = val_feeder.input_index_generate_batch()
        val_feed = {g.inputs: val_inputs,
                    g.labels: val_labels,
                    g.seq_len: val_seq_len,
                    g.keep_drop: 1.0,
                    g.rnn_keep_drop: 1.0}

        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx = np.random.permutation(num_train_samples)
            train_cost = train_err = 0
            start_time = time.time()
            batch_time = time.time()
            # the tracing part
            for cur_batch in range(num_batches_per_epoch):
                if (cur_batch ) % 100 == 0:
                    print('batch', cur_batch, ': time', time.time() - batch_time)
                batch_time = time.time()
                indexs = [shuffle_idx[i % num_train_samples] for i in
                          range(cur_batch * FLAGS.batch_size, (cur_batch + 1) * FLAGS.batch_size)]
                batch_inputs, batch_seq_len, batch_labels = train_feeder.input_index_generate_batch(indexs)
                # batch_inputs,batch_seq_len,batch_labels=utils.gen_batch(FLAGS.batch_size)
                feed = {g.inputs: batch_inputs,
                        g.labels: batch_labels,
                        g.seq_len: batch_seq_len,
                        g.keep_drop: 0.5,
                        g.rnn_keep_drop: 0.8}
                # print(cur_batch)
                # print(batch_seq_len)
                # _,batch_cost, the_err,d,lr,train_summary,step = sess.run([optimizer,cost,lerr,decoded[0],learning_rate,merged_summay,global_step],feed)
                # _,batch_cost, the_err,d,lr,step = sess.run([optimizer,cost,lerr,decoded[0],learning_rate,global_step],feed)
                # the_err,d,lr = sess.run([lerr,decoded[0],learning_rate])

                # if summary is needed
                # batch_cost,step,train_summary,_ = sess.run([cost,global_step,merged_summay,optimizer],feed)

                # if summary is needed
                summary_str, batch_cost, step, _ = sess.run([g.merged_summay, g.cost, g.global_step, g.optimizer], feed)
                # batch_cost, step, _ = sess.run([g.cost, g.global_step, g.optimizer], feed)
                # print(step)
                # calculate the cost
                train_cost += batch_cost * FLAGS.batch_size

                # # the tracing part
                # _,batch_cost,the_err,step,lr,d = sess.run([optimizer,cost,lerr,
                #    global_step,learning_rate,decoded[0]],feed)
                # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                # run_metadata=run_metadata)
                # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                # race_file.write(trace.generate_chrome_trace_format())
                # trace_file.close()

                # 可以暂时不写
                train_writer.add_summary(summary_str, step)

                # save the checkpoint
                if step % FLAGS.save_steps == 1:
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    logger.info('save the checkpoint of{0}', format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step=step)
                # train_err+=the_err*FLAGS.batch_size
                # do validation
                if step % FLAGS.validation_steps == 0:
                    dense_decoded, lastbatch_err, lr = sess.run([g.dense_decoded, g.lerr,
                                                                 g.learning_rate], val_feed)
                    # print the decode result
                    acc = utils.accuracy_calculation2(val_feeder.image_names, val_feeder.labels, dense_decoded,
                                                      ignore_value=-1, isPrint=True)
                    avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)
                    # train_err/=num_train_samples
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{} Epoch {}/{}, accuracy = {:.3f},avg_train_cost = {:.3f}, " \
                          "lastbatch_err = {:.3f}, time = {:.3f},lr={:.8f}"
                    print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                     cur_epoch + 1, FLAGS.num_epochs, acc, avg_train_cost, lastbatch_err,
                                     time.time() - start_time, lr))


def test(test_dir=None):
    g = den_CNN_BN_BiLSTM_LSTM_CTC(mode='test')

    # with g.graph.as_default():
    print('loading test data, please wait---------------------', end=' ')
    test_feeder = utils.DataIterator(data_dir=test_dir)
    print('get image: ', test_feeder.size)

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)
    with tf.Session(graph=g.graph, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        g.graph.finalize()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            # the global_step will restore as well
            saver.restore(sess, ckpt)
            print('restore from the checkpoint{0}'.format(ckpt))

        print('=============================begin testing=============================')
        start_time = time.time()
        test_inputs, test_seq_len, test_labels = test_feeder.input_index_generate_batch()
        print(test_inputs.shape)
        test_feed = {g.inputs: test_inputs,
                     g.labels: test_labels,
                     g.seq_len: test_seq_len,
                     g.keep_drop: 1.0,
                     g.rnn_keep_drop: 1.0}

        dense_decoded, lastbatch_err, lr = sess.run([g.dense_decoded, g.lerr,
                                                     g.learning_rate], test_feed)
        # print the decode result
        acc = utils.accuracy_calculation2(test_feeder.image_names, test_feeder.labels, dense_decoded,
                                          ignore_value=-1, isPrint=True)
        now = datetime.datetime.now()
        log = "{}/{} {}:{}:{} accuracy = {:.3f}, lastbatch_err = {:.3f}, lr={:.8f}"
        print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                         acc, lastbatch_err, lr))


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        train(train_dir='ch_train', val_dir='ch_test')
        # test(test_dir='ch_test')

    # cv2.waitKey()
    # cv2.destroyAllWindows()