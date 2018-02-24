import tensorflow as tf
import cv2
import numpy as np
import detect_face
from ProcessImage import processImg


# 人脸检测(FD)的参数
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction = 1.0

# 建立两个模型，人脸检测(FD)和人脸识别(FR)
graph_FD = tf.Graph()
graph_FR = tf.Graph()

# sess_FD = tf.Session(graph=graph_FD, config=tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=True))
# sess_FR = tf.Session(graph=graph_FR)

print('Creating networks and loading parameters')

# 加载FD模型
with graph_FD.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess_FD = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=True))
    with sess_FD.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess_FD, None)

# 加载FR模型
with graph_FR.as_default():
    sess_FR = tf.Session(graph=graph_FR)
    with sess_FR.as_default():
        saver = tf.train.import_meta_graph('./model/train_FR.ckpt.meta')
        saver.restore(sess_FR, './model/train_FR.ckpt')

        graph = tf.get_default_graph()

        input_x = graph.get_tensor_by_name('input_x:0')
        label_y = graph.get_tensor_by_name('label_y:0')
        drop_prob = graph.get_tensor_by_name('drop_prob:0')

        prediction = graph.get_collection('prediction')[0]

        result = tf.argmax(prediction, 1)

cap = cv2.VideoCapture(0)
count = 0

while True:
    # 读取摄像头画面
    ret, frame = cap.read()

    # 每5帧进行一次
    if count % 1 == 0:
        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        numof_faces = bounding_boxes.shape[0]  # 人脸数目
        print('找到人脸数目为：{}'.format(numof_faces))

        for face_position in bounding_boxes:
            face_position = face_position.astype(int)
            # 截取面部区域
            faceFrame = frame[face_position[1]:face_position[3], face_position[0]:face_position[2]]

            # 处理面部区域
            faceFrame = processImg(faceFrame)
            faceFrame = np.array(faceFrame)
            faceFrame = faceFrame.reshape(1, 64, 64, 3)

            res = sess_FR.run(result, feed_dict={input_x: faceFrame, drop_prob: 0.0})

            if res[0] == 1:
                name = 'Wentao'
            elif res[0] == 0:
                name = 'Other'
            else:
                name = ''
            cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]),
                          (0, 255, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (face_position[0] + 5, face_position[1] + 5), font, 1.0,
                        (0, 255, 0), 2)

    # 显示frame
    cv2.imshow('frame', frame)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sess_FD.close()
sess_FR.close()

# ============test==============
# import os
# from sklearn.model_selection import train_test_split
#
# size = 64
#
# my_faces_path = './faceData/Wentao/'
# other_faces_path = './faceData/Others/'
#
# imgs = []
# labels = []
#
# def readData(path):
#     for filename in os.listdir(path):
#         if filename.endswith('.jpg'):
#             filename = path + filename
#             img = cv2.imread(filename)
#             # top, bottom, left, right = getPaddingSize(img)
#             #
#             # img = cv2.copyMakeBorder(img, top, bottom, left, right,
#             #                          cv2.BORDER_CONSTANT,
#             #                          value=[0, 0, 0])
#             # img = cv2.resize(img, (h, w))
#             img = processImg(img)
#             imgs.append(img)
#             labels.append(path)
#
#
# # 读取图片数据
# readData(my_faces_path)
# readData(other_faces_path)
#
# # 将img数据和label数据转换为数组
# imgs = np.array(imgs)
# labels = np.array([[0, 1] if label == my_faces_path else [1, 0] for label in labels])
#
# # 划分测试集合训练集
# train_x, test_x, train_y, test_y = train_test_split(imgs, labels, test_size=0.05)
#
# # reshape input
# train_x = train_x.reshape(train_x.shape[0], size, size, 3)
# test_x = test_x.reshape(test_x.shape[0], size, size, 3)
#
# # normalize
# train_x = train_x.astype('float32')/255.0
# test_x = test_x.astype('float32')/255.0
#
# # 初始化模型
# sess_FR = tf.Session()
# saver = tf.train.import_meta_graph('./model/train_FR.ckpt.meta')
# saver.restore(sess_FR, './model/train_FR.ckpt')
#
# graph = tf.get_default_graph()
#
# input_x = graph.get_tensor_by_name('input_x:0')
# label_y = graph.get_tensor_by_name('label_y:0')
# drop_prob = graph.get_tensor_by_name('drop_prob:0')
#
# prediction = graph.get_collection('prediction')[0]
#
# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(label_y, 1)), tf.float32))
# res = sess_FR.run(accuracy, feed_dict={input_x: test_x, label_y: test_y, drop_prob: 0.0})
#
# print(res)
