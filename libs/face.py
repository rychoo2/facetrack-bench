import datetime

import dlib
import cv2
from imutils import face_utils
from libs.eye_landmark import detect_pupil
import os
import numpy as np

curdir = os.path.dirname(os.path.realpath(__file__))
landmarks_predictor_path = curdir + "/../../data/shape_predictor_68_face_landmarks.dat"
eyes_cascade_path = curdir + '/../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
face_cascade_path = curdir + '/../../data/haarcascades/haarcascade_frontalface_alt2.xml'
cv_facemarks_path = curdir + '/../../data/lbfmodel.yaml'

facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(cv2.samples.findFile(cv_facemarks_path))

face_detector = dlib.get_frontal_face_detector()
landmarks_detector = dlib.shape_predictor(landmarks_predictor_path)
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile(face_cascade_path))


def get_face(img):
    frame_processed = img.copy()
    frame_processed = cv2.medianBlur(frame_processed, 5)
    frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)
    frame_processed = cv2.equalizeHist(frame_processed)

    frame_final = frame_processed
    faces_cv2 = face_cascade.detectMultiScale(frame_final, minSize=(int(0.1 * frame_final.shape[0]), int(0.1 * frame_final.shape[1])))
    primary_face_csv2_index, primary_face_csv2 = get_largest_shape([dict(x=x, y=y, width=w, height=h) for (x, y, w, h) in faces_cv2])
    faces_dlib = face_detector(frame_final, 1)
    primary_face_dlib_index, primary_face_dlib = get_largest_shape(
        [dict(x=rect.left(), y=rect.top(), width=rect.width(), height=rect.height()) for rect in faces_dlib])

    face = dict(bbox_dlib=[[primary_face_dlib['x'], primary_face_dlib['y']],
                      [primary_face_dlib['x'] + primary_face_dlib['width'],
                       primary_face_dlib['y'] + primary_face_dlib['height']]] if primary_face_dlib else [[None, None], [None, None]],
                bbox_opencv=[[primary_face_csv2['x'], primary_face_csv2['y']],
                       [primary_face_csv2['x'] + primary_face_csv2['width'],
                        primary_face_csv2['y'] + primary_face_csv2['height']]] if primary_face_csv2 else [[None, None], [None, None]],
                landmarks_dlib=[],
                landmarks_opencv=[],
                left_eye=dict(bbox=[[None, None], [None, None]]),
                right_eye=dict(bbox=[[None, None], [None, None]]))

    if None not in face['bbox_dlib'][0]:

        _shape = landmarks_detector(frame_final, dlib.rectangle(face['bbox_dlib'][0][0], face['bbox_dlib'][0][1],
                                                                face['bbox_dlib'][1][0],
                                                                face['bbox_dlib'][1][1]))

        shape = face_utils.shape_to_np(_shape)
        face['landmarks_dlib'] = shape.tolist()

    if None not in face['bbox_opencv'][0]:
        ok, landmarks2 = facemark.fit(frame_final, faces=faces_cv2[primary_face_csv2_index: primary_face_csv2_index+1])
        #here is the bug taking first landmars rather than largest
        face['landmarks_opencv'] = [[int(x[0]), int(x[1])] for x in landmarks2[0][0]]

    face_landmarks = face['landmarks_dlib'] or face['landmarks_opencv']

    if len(face_landmarks) > 0:
        for (eye_name, eye_bb) in [('right_eye', cv2.boundingRect(np.array(face_landmarks[36: 41]))),
                                   ('left_eye', cv2.boundingRect(np.array(face_landmarks[42: 47])))]:
            eyebb_y1, eyebb_y2, eyebb_x1, eyebb_x2 = eye_bb[1] - int(0.8 * eye_bb[3]), \
                                                     eye_bb[1] + int(1.8 * eye_bb[3]), \
                                                     eye_bb[0] - int(0.8 * eye_bb[2]), \
                                                     eye_bb[0] + int(1.8 * eye_bb[2])

            eye1_frame = frame_final[eyebb_y1: eyebb_y2, eyebb_x1:eyebb_x2]
            eye1_frame = cv2.cvtColor(eye1_frame, cv2.COLOR_GRAY2RGB)

            face[eye_name]['bbox'] = [[eyebb_x1, eyebb_y1], [eyebb_x2, eyebb_y2]]

            pupil = detect_pupil(eye1_frame)
            if pupil:
                face[eye_name]['pupil'] = [eyebb_x1 + round(pupil.pt[0]), eyebb_y1 + round(pupil.pt[1])]

    return face

def generate_landmark_image(input_img, face):
    output_img = input_img.copy()
    if None not in face['bbox_dlib'][0]:
        cv2.rectangle(output_img, pt1= tuple(face['bbox_dlib'][0]), pt2=tuple(face['bbox_dlib'][1]), color=(0, 0, 255))
    for (x, y) in face['landmarks_dlib']:
        cv2.circle(output_img, (round(x), round(y)), 2, (0, 0, 255), -1)
    if None not in face['bbox_opencv'][0]:
        cv2.rectangle(output_img, pt1=tuple(face['bbox_opencv'][0]), pt2=tuple(face['bbox_opencv'][1]), color=(255, 0, 0))
    for (x, y) in face['landmarks_opencv']:
        cv2.circle(output_img, (round(x), round(y)), 2, (255, 0, 0), -1)
    for eye in ['left_eye', 'right_eye']:
        if 'bbox' in face[eye] and face[eye]['bbox'][0][0]:
            cv2.rectangle(output_img, pt1=tuple(face[eye]['bbox'][0]), pt2=tuple(face[eye]['bbox'][1]),
                      color=(255, 255, 0))
        pupil = face[eye].get('pupil')
        if pupil:
            cv2.circle(output_img, tuple(pupil), 1, (0, 255, 0), 2)
    return output_img

def get_largest_shape(shapes):
    max_width = 0
    max_shape = None
    max_index = None
    for index, shape in enumerate(shapes):
        if shape['width'] > max_width:
            max_width = shape['width']
            max_shape = shape
            max_index = index
    return max_index, max_shape
