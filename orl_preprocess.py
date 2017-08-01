# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import cv2
import numpy as np
from PIL import Image
import random

input_path = "./orl"
train_path = "./train"
test_path = "./test"

if not os.path.exists(train_path):
    os.mkdir(train_path)

if not os.path.exists(test_path):
    os.mkdir(test_path)

for i in range(1, 41):
    if not os.path.exists(train_path + '/' + str(i)):
        os.mkdir(train_path + '/' + str(i))
    if not os.path.exists(test_path + '/' + str(i)):
        os.mkdir(test_path + '/' + str(i))


# 生成训练和测试的数据
def generate_data(train_path, test_path):
    index = 1
    output_index = 1
    for (dirpath, dirnames, filenames) in os.walk(input_path):
        # 打乱文件列表，相当于是随机选取8张训练集，2张测试
        random.shuffle(filenames)
        for filename in filenames:
            if filename.endswith('.bmp'):
                img_path = dirpath + '/' + filename
                # 使用opencv 读取图片
                img_data = cv2.imread(img_path)
                # 按照论文中的将图片大小调整为28 * 28
                img_data = cv2.resize(img_data, (28, 28), interpolation=cv2.INTER_AREA)
                if index < 3:
                    cv2.imwrite(test_path + '/' + str(output_index) + '/' + str(index) + '.jpg', img_data)
                    index += 1
                elif 10 >= index >= 3:
                    cv2.imwrite(train_path + '/' + str(output_index) + '/' + str(index) + '.jpg', img_data)
                    index += 1
                if index > 10:
                    output_index += 1
                    index = 1

generate_data(train_path, test_path)

# def get_data():
#     input_data = []
#     input_label = []
#     # 显示tfrecord格式的图片
#     filename_queue = tf.train.string_input_producer(["orl_faces.tfrecords"])
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw': tf.FixedLenFeature([], tf.string),
#                                        })
#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     # img = tf.reshape(img, [28, 28, 1])
#     label = tf.cast(features['label'], tf.int32)
#     with tf.Session() as sess:
#         init_op = tf.global_variables_initializer()
#         sess.run(init_op)
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#         for i in range(400):
#             example, l = sess.run([img, label])
#             # example = np.array(example)
#             a = int(i / 10)
#             b = i % 10
#             example = np.array(example) / 255.0
#             # example.reshape([28, 28, 1])
#             input_data.append(example)
#
#             tmp = np.zeros(10)
#             tmp[int(l) % 10] = 1
#             input_label.append(tmp)
#
#         input_data = np.array(input_data)
#         input_label = np.array(input_label)
#         coord.request_stop()
#         coord.join(threads)
#         return input_data, input_label
#

