# -*- coding:utf-8 -*-
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import orl_inference
import cv2


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串类型
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


train_path = "./train/"
test_path = "./test/"
classes = {i: i for i in range(1, 41)}
writer_train = tf.python_io.TFRecordWriter("orl_train.tfrecords")
writer_test = tf.python_io.TFRecordWriter("orl_test.tfrecords")


def generate():
    # 遍历字典
    for index, name in enumerate(classes):
        train = train_path + str(name) + '/'
        test = test_path + str(name) + '/'
        for img_name in os.listdir(train):
            img_path = train + img_name  # 每一个图片的地址
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(index + 1),
                'img_raw': _bytes_feature(img_raw)
            }))
            writer_train.write(example.SerializeToString())
        for img_name in os.listdir(test):
            img_path = test + img_name  # 每一个图片的地址
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(index + 1),
                'img_raw': _bytes_feature(img_raw)
            }))
            writer_test.write(example.SerializeToString())
    writer_test.close()
    writer_train.close()

generate()


# def read_and_decode(filename):
#     # 生成一个队列
#     filename_queue = tf.train.string_input_producer([filename])
#
#     reader = tf.TFRecordReader()
#     # 返回文件名和文件
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw': tf.FixedLenFeature([], tf.string),
#                                        })
#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [28, 28, 3])
#     img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
#     label = tf.cast(features['label'], tf.int32)
#     return img, label


