# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2
import orl_inference
from sklearn.model_selection import train_test_split
import random
import orl_preprocess

SIZE = 28
# 配置CNN参数
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
LEARNING_RATE_BASE = 0.1  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习的衰减率
BATCH_SIZE = 20

MODEL_SAVE_PATH = "/path/to/model/"
MODEL_NAME = "model.ckpt"


# # 输入数据路径
# input_path = "./data"
#
# # 输入数据
# input_data = [[0 for i in range(10)] for j in range(40)]
# input_label = []
#
#
# def read(path):
#     index = 1
#     sort_index = 0
#     for (dirpath, dirnames, filenames) in os.walk(input_path):
#         for filename in filenames:
#             if filename.endswith('.jpg'):
#                 img_path = dirpath + '/' + filename
#                 img_data = cv2.imread(img_path)
#                 img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
#                 if index % 10 == 0:
#                     input_label.append(dirpath)
#                     input_data[sort_index][0] = img_data
#                     sort_index += 1
#                 else:
#                     input_data[sort_index][index % 10] = img_data
#                 index += 1
#
#
# read(input_path)
# cv2.imshow("2", input_data[1][0])
# # 将图片数据与标签转化为数组
# input_data = np.array(input_data)
# # print(input_label)
# for i in range(len(input_label)):
#     tmp = input_label[i].split("\\")[1]
#     label = np.zeros(40)
#     label[int(tmp) - 1] = 1
#     input_label[i] = label
#
# train_x = []
# test_x = []
# train_y = []
# test_y = []
#
# # 形成训练集和测试集
# for i in range(40):
#     tn_x, tt_x, tn_y, tt_y = train_test_split(input_data[i], input_label[i], test_size=0.2,
#                                               random_state=random.randint(0, 100))
#     train_x.append(tn_x)
#     test_x.append(tt_x)
#     train_y.append(tn_y)
#     test_y.append(tt_y)

def get_label(label):
    ys = []
    for i in range(label.size):
        tmp = np.zeros(40)
        tmp[label[i] - 1] = 1
        ys.append(tmp)
    return ys


def train(data, label):
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE, SIZE, SIZE, orl_inference.NUM_CHANNELS],
                       name='x-input')

    y_ = tf.placeholder(tf.float32, [None, orl_inference.OUTPUT_NODE], name='y-output')

    # 使用L2正则化计算损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    image_batch, label_batch = tf.train.shuffle_batch(
        [data, label], batch_size=BATCH_SIZE,
        capacity=capacity, min_after_dequeue=min_after_dequeue
    )

    y = orl_inference.inference(x, False, regularizer)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算交叉熵作为刻画预测值和真实值之间的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

    # 计算所有样例中交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        320 / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
        .minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()

    # 验证
    # accuracy = tf.reduce_mean()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # 迭代的训练网络
        for i in range(TRAINING_STEPS):
            xs, ys = sess.run([image_batch, label_batch])
            xs = xs / 255.0
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          SIZE,
                                          SIZE,
                                          orl_inference.NUM_CHANNELS))
            # 将图像和标签数据通过tf.train.shuffle_batch整理成训练时需要的batch
            ys = get_label(ys)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})

            if i % 100 == 0:
                # 每10轮输出一次在训练集上的测试结果
                acc = loss.eval({x: reshaped_xs, y_: ys})
                print("After %d training step[s], loss on training"
                      " batch is %g. " % (step, loss_value))

                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step
                )
                # logit = orl_inference.inference(image_batch)
        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    # 显示tfrecord格式的图片
    filename_queue = tf.train.string_input_producer(["orl_train.tfrecords"])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    label = tf.cast(features['label'], tf.int32)
    train(img, label)


if __name__ == '__main__':
    tf.app.run()
