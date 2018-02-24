import tensorflow as tf
import detect_face
import cv2
import numpy as np
from ProcessImage import processImg

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

img_path = './faceData/Wentao/Wentao-123.jpg'
img_path_2 = './faceData/Others/Others-4000.jpg'
imgs = []

img = cv2.imread(img_path)
img = processImg(img)
imgs.append(img)

imgs = np.array(imgs)


sess_FR = tf.Session()
saver = tf.train.import_meta_graph('./model/train_FR.ckpt.meta')
saver.restore(sess_FR, './model/train_FR.ckpt')

graph = tf.get_default_graph()

input_x = graph.get_tensor_by_name('input_x:0')
drop_prob_1 = graph.get_tensor_by_name('drop_prob_1:0')
drop_prob_2 = graph.get_tensor_by_name('drop_prob_2:0')

prediction = graph.get_collection('prediction')[0]

result = tf.argmax(prediction, 1)

res = sess_FR.run(result, feed_dict={input_x: imgs, drop_prob_1: 0.0, drop_prob_2: 0.0})

print(res)
