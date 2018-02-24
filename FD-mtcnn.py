import tensorflow as tf
import detect_face
import cv2
import os

output_dir = 'faceData/Wentao/'

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

cap = cv2.VideoCapture(0)
# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml"q)
count = 0
count2 = 3368
while count2 < 5000:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if count % 5 == 0:
        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        numof_faces = bounding_boxes.shape[0]  # 人脸数目
        print('找到人脸数目为：{}'.format(numof_faces))

        for face_position in bounding_boxes:
            face_position = face_position.astype(int)

            w = face_position[2] - face_position[0]
            h = face_position[3] - face_position[1]

            if w > 100 and h > 100:
                print('save')
                count2 += 1
                count2str = str(count2)
                faceFrame = frame[face_position[1]:face_position[3], face_position[0]:face_position[2]]
                cv2.imwrite(output_dir + 'Wentao-'+count2str+'.jpg', faceFrame)

            cv2.rectangle(frame, (face_position[0], face_position[1]), (face_position[2], face_position[3]),
                          (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()