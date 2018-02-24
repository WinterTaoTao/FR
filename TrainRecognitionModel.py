import tensorflow as tf
import cv2
import os
import numpy as np
import random
import sys
from sklearn.model_selection import train_test_split

my_faces_path = './faceData/Wentao/'
other_faces_path = './faceData/Others/'

size = 64

imgs = []
labels = []


def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)

    if w < longest:
        border = longest - w
        left = border // 2
        right = border - left

    elif h < longest:
        border = longest - h
        top = border // 2
        bottom = border - top

    else:
        pass

    return top, bottom, left, right


def readData(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + filename
            img = cv2.imread(filename)

            top, bottom, left, right = getPaddingSize(img)

            img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT,
                                     value=[0, 0, 0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labels.append(path)


readData(my_faces_path)
readData(other_faces_path)

# 将img数据和label数据转换为数组
imgs = np.array(imgs)
labels = np.array([[0, 1] if label == my_faces_path else [1, 0] for label in labels])

# 划分测试集合训练集
train_x, test_x, train_y, test_y = train_test_split(imgs, labels, test_size=0.05)

# reshape input
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

# normalize
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

# batch size = 100
batch_size = 100
num_batch = len(train_x) // batch_size

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)


def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def dropout(x, keep):
    return tf.nn.dropout(x, keep)


def cnnLayer():
    # 第一层
    W1 = weightVariable([3, 3, 3, 32])
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # dropout
    drop1 = dropout(pool1, keep_prob_5)

    # conv1 = tf.layers.conv2d(
    #     inputs=x,
    #     filters=32,
    #     kernel_size=3,
    #     strides=1,
    #     padding='same',
    #     activation=tf.nn.relu,
    #     kernel_initializer=tf.random_normal_initializer(stddev=0.01),
    #     bias_initializer=tf.random_normal_initializer(stddev=1)
    # )
    #
    # pool1 = tf.layers.max_pooling2d(
    #     inputs=conv1,
    #     pool_size=2,
    #     strides=2,
    #     padding='same'
    # )
    #
    # drop1 = tf.layers.dropout(pool1, rate=keep_prob_5)

    # 第二层
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    Wf = weightVariable([8 * 16 * 32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8 * 16 * 32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512, 2])
    bout = weightVariable([2])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

def cnnTrain():
    out = cnnLayer()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))

    # 保存loss和accuracy供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())

        for n in range(10):
            for i in range(num_batch):
                batch_x = train_x[i * batch_size : (i + 1) * batch_size]
                batch_y = train_y[i * batch_size : (i + 1) * batch_size]

                # 训练返回三个数据
                _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                            feed_dict={x: batch_x, y_: batch_y, keep_prob_5: 0.5, keep_prob_75: 0.75})
                summary_writer.add_summary(summary, n*num_batch+i)

                print(n*num_batch+i, loss)

                if(n*num_batch + i) % 100 == 0:
                    acc = accuracy.eval({x: test_x, y_: test_y, keep_prob_5:1.0, keep_prob_75:1.0})
                    print(n*num_batch + i, acc)
                    # 准确率大雨0.98时保存并推出
                    if acc > 0.98 and n > 2:
                        saver.save(sess, './model/train_FR.model', global_step=n*num_batch+i)
                        sys.exit(0)

        print('accuracy less than 0.98!')

cnnTrain()