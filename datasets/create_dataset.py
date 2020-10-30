# -*- coding: utf-8 -*-

import os
import glob
import tensorflow as tf  
import numpy as np


def parse_tfrecords(fileList, batch_size):
    file_queue = tf.compat.v1.train.string_input_producer(fileList)
    reader = tf.compat.v1.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.compat.v1.parse_single_example(
                                                serialized_example,
                                                features={
                                                    'data':  tf.compat.v1.FixedLenFeature([], tf.string),                
                                                    'label': tf.compat.v1.FixedLenFeature([], tf.int64)
                                                })
    data = tf.compat.v1.decode_raw(features['data'], tf.float32)
    data = tf.reshape(data, (313*128,))
    # data = tf.expand_dims(data, axis=0)
    label = features['label']
    # label = tf.expand_dims(label, axis=0)
    
    data_batch, label_batch = tf.compat.v1.train.shuffle_batch([data, label],
                                                                batch_size=batch_size,
                                                                capacity=8000,
                                                                num_threads=4,
                                                                min_after_dequeue=2000)
    return data_batch, label_batch


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    tfDirs = r"E:\project\anormal\C2AE\tfrecords"
    fileList = glob.glob(os.path.join(tfDirs, "*.tfrecord"))
    data_b, label_b = parse_tfrecords(fileList, 4)
    label_bb = tf.cast(label_b, tf.int32)
    mostIndex = tf.argmax(tf.compat.v1.bincount(label_bb))
    mostValue = tf.gather(label_bb, mostIndex)
    label_binary = tf.cast(tf.equal(label_bb, mostValue), tf.float32)
    label_train = tf.where(tf.greater(label_binary, 0), -1, 1)
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(coord=coord)
        for i in range(12):
            datas, labels, labels_t = sess.run([data_b, label_b, label_train])
            print(labels)
            print(labels_t)
            print("------------------")
        coord.request_stop()
        coord.join(threads)

    maxIndex = tf.argmax(tf.compat.v1.bincount(label_b))


