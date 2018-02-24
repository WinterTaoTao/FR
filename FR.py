import tensorflow as tf
import detect_face
import os
import cv2
import numpy as np
from ProcessImage import processImg
from sklearn.model_selection import train_test_split


# # 人脸检测(FD)的参数
# minsize = 20  # minimum size of face
# threshold = [0.6, 0.7, 0.7]  # three steps's threshold
# factor = 0.709  # scale factor
# gpu_memory_fraction = 1.0
#
# # 建立两个模型，人脸检测(FD)和人脸识别(FR)
# graph_FD = tf.Graph()
# graph_FR = tf.Graph()
#
# sess_FD = tf.Session(graph=graph_FD, config=tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=True))
# sess_FR = tf.Session(graph=graph_FR)
#
# print('Creating networks and loading parameters')
#
# # 加载FD模型
# with graph_FD.as_default():
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
#     with sess_FD.as_default():
#         pnet, rnet, onet = detect_face.create_mtcnn(sess_FD, None)
#
# # 加载FR模型
# with graph_FR.as_default():
#     with sess_FR.as_default():
#
#
# cap = cv2.VideoCapture(0)
# count = 0

# ===========================

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


sess_FR = tf.Session()
saver = tf.train.import_meta_graph('./model/train_FR.ckpt.meta')
saver.restore(sess_FR, './model/train_FR.ckpt')

graph = tf.get_default_graph()

input_x = graph.get_tensor_by_name('input_x:0')
label_y = graph.get_tensor_by_name('label_y:0')
drop_prob_1 = graph.get_tensor_by_name('drop_prob_1:0')
drop_prob_2 = graph.get_tensor_by_name('drop_prob_2:0')

prediction = graph.get_collection('prediction')[0]

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(label_y, 1)), tf.float32))

# result = tf.argmax(prediction, 1)

res = sess_FR.run(accuracy, feed_dict={input_x: imgs[5000:6000], label_y: labels[5000:6000], drop_prob_1: 0.0, drop_prob_2: 0.0})

print(res)
