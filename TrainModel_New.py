import tensorflow as tf
import cv2
import os
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from ProcessImage import processImg

size = 64

my_faces_path = './faceData/Wentao/'
other_faces_path = './faceData/Others/'

imgs = []
labels = []

def readData(path):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + filename
            img = cv2.imread(filename)
            # top, bottom, left, right = getPaddingSize(img)
            #
            # img = cv2.copyMakeBorder(img, top, bottom, left, right,
            #                          cv2.BORDER_CONSTANT,
            #                          value=[0, 0, 0])
            # img = cv2.resize(img, (h, w))
            img = processImg(img)
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

x = tf.placeholder(tf.float32, [None, size, size, 3], name='input_x')
y_ = tf.placeholder(tf.float32, [None, 2], name='label_y')

drop_prob = tf.placeholder(tf.float32, name='drop_prob')
# drop_prob = tf.placeholder(tf.float32, name='drop_prob_2')


def conv2d(inputs, filters, kernel_size):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        bias_initializer=tf.constant_initializer(0.1)
    )


def maxPool(inputs):
    return tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=2,
        strides=2,
        padding='same'
    )


def dropout(inputs, drop_prob):
    return tf.layers.dropout(inputs=inputs, rate=drop_prob)


def cnnLayers():
    # 第一层
    conv1 = conv2d(inputs=x, filters=32, kernel_size=3)
    batch_nor1 = tf.layers.batch_normalization(conv1)
    active1 = tf.nn.relu(batch_nor1)
    pool1 = maxPool(active1)
    drop1 = dropout(pool1, drop_prob=drop_prob)

    # 第二层
    conv2 = conv2d(inputs=drop1, filters=64, kernel_size=3)
    batch_nor2 = tf.layers.batch_normalization(conv2)
    active2 = tf.nn.relu(batch_nor2)
    pool2 = maxPool(active2)
    drop2 = dropout(pool2, drop_prob=drop_prob)

    # 第三层
    conv3 = conv2d(inputs=drop2, filters=64, kernel_size=3)
    batch_nor3 = tf.layers.batch_normalization(conv3)
    active3 = tf.nn.relu(batch_nor3)
    pool3 = maxPool(active3)
    drop3 = dropout(pool3, drop_prob=drop_prob)

    # 全连接层
    drop3_flat = tf.layers.flatten(drop3)
    fc = tf.layers.dense(
        inputs=drop3_flat,
        units=512,
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        bias_initializer=tf.constant_initializer(0.1)
    )
    drop_fc = dropout(fc, drop_prob=drop_prob)

    # 输出层
    prediction = tf.layers.dense(
        inputs=drop_fc,
        units=2,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
        bias_initializer=tf.constant_initializer(0.1)
    )

    return prediction


def cnnTrain():
    prediction = cnnLayers()
    tf.add_to_collection('prediction', prediction)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1)), tf.float32))

    # 保存loss和accuracy供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())

        for n in range(10):
            for i in range(num_batch):
                batch_x = train_x[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]

                _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                            feed_dict={x: batch_x, y_: batch_y, drop_prob: 0.5})
                summary_writer.add_summary(summary, n*num_batch+i)

                print(n*num_batch+i, loss)

                if(n*num_batch + i) % 100 == 0:
                    acc = accuracy.eval({x: test_x, y_: test_y, drop_prob: 0.0})
                    print('acciracy:', n*num_batch + i, acc)
                    # 准确率大于0.98时保存并退出
                    if acc > 0.98 and n > 2:
                        saver.save(sess, './model/train_FR.ckpt')
                        sys.exit(0)

        print('accuracy less than 0.98!')


cnnTrain()
