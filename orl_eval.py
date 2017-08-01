# -*- coding:utf-8 -*-
import time
import tensorflow as tf
import numpy as np
import orl_inference
import orl_train
import cv2
# 每一秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 1


def get_label(label):
    ys = []
    for i in range(label.size):
        tmp = np.zeros(40)
        tmp[label[i] - 1] = 1
        ys.append(tmp)
    return ys


def evaluate():
    with tf.Graph().as_default() as g:
        filename_queue = tf.train.string_input_producer(["orl_test.tfrecords"])
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
        min_after_dequeue = 100
        capacity = min_after_dequeue + 3 * 200
        image_batch, label_batch = tf.train.shuffle_batch(
            [img, label], batch_size=80,
            capacity=capacity, min_after_dequeue=min_after_dequeue
        )

        x = tf.placeholder(tf.float32,
                           [80,
                            orl_inference.IMAGE_SIZE,
                            orl_inference.IMAGE_SIZE,
                            orl_inference.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, orl_inference.OUTPUT_NODE], name='y-input'
        )

        y = orl_inference.inference(x, None, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(
            orl_train.MOVING_AVERAGE_DECAY
        )
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        # 每隔EVAL_INTERVAL_SECS秒调用一次
        while True:
            with tf.Session() as sess:
                test = cv2.imread('./data/20/10.jpg')
                test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
                test = np.array(test)
                test = test / 255.0
                test_re = np.reshape(test, (1, 28, 28, 1))

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                xs, ys = sess.run([image_batch, label_batch])
                ys = get_label(ys)
                xs = xs / 255.0
                validate_feed = {x: xs,
                                 y_: ys}

                cpkt = tf.train.get_checkpoint_state(
                    orl_train.MODEL_SAVE_PATH
                )
                if cpkt and cpkt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, cpkt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = cpkt.model_checkpoint_path \
                        .split('/')[-1].split('-')[-1]
                    # result = sess.run(y, feed_dict={x: test_re})
                    # re = np.where(result == np.max(result))
                    # ss = tf.argmax(result, 1)
                    # tt = np.argmax(result, 1)
                    # print('result is %d'%(tt[0] + 1))
                    # # print('hehe')
                    accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                    print("After %s training steps, validation "
                          "accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()

