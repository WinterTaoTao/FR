import tensorflow as tf
import detect_face
import sys
import os
import cv2

input_dir = './faceData/input_img/'
output_dir = './faceData/Others/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction=1.0

print('Creating networks and loading parameters')

with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}, log_device_placement=True))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

index = 1
for path, dirnames, filenames in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            img_path = path+'/'+filename
            img = cv2.imread(img_path)
            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            numof_faces = bounding_boxes.shape[0]
            print('找到人脸数目为：{}'.format(numof_faces))

            for face_position in bounding_boxes:
                face_position = face_position.astype(int)

                w = face_position[2] - face_position[0]
                h = face_position[3] - face_position[1]

                print('save')
                faceImg = img[face_position[1]:face_position[3], face_position[0]:face_position[2]]
                indexStr = str(index)
                cv2.imwrite(output_dir + 'Others-' + indexStr + '.jpg', faceImg)
                index += 1