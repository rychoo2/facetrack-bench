import datetime

import dlib
import cv2
from imutils import face_utils
from libs.simple_eye_landmark_detector import SimpleEyeLandmarkDetector
import os
import numpy as np
from ext.detector.face_detector import MTCNNFaceDetector
from keras import backend as K
from ext.elg_keras import KerasELG

curdir = os.path.dirname(os.path.realpath(__file__))
landmarks_predictor_path = curdir + "/../../data/shape_predictor_68_face_landmarks.dat"
eyes_cascade_path = curdir + '/../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
face_cascade_path = curdir + '/../../data/haarcascades/haarcascade_frontalface_alt2.xml'
cv_facemarks_path = curdir + '/../../data/lbfmodel.yaml'
mtcnn_weights_dir = "../../data/mtcnn_weights/"
elg_weights_dir = "../../data/elg_weights/elg_keras.h5"
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(cv2.samples.findFile(cv_facemarks_path))

face_detector = dlib.get_frontal_face_detector()
landmarks_detector = dlib.shape_predictor(landmarks_predictor_path)
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile(face_cascade_path))
eye_landmark_detector = SimpleEyeLandmarkDetector()

fd_mtcnn = MTCNNFaceDetector(sess=K.get_session(), model_path=mtcnn_weights_dir)

elg_model = KerasELG()
elg_model.net.load_weights(elg_weights_dir)

def get_face(img):
    frame_processed = img.copy()

    # frame_processed = cv2.medianBlur(frame_processed, 5)
    # frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)
    # frame_processed = cv2.equalizeHist(frame_processed)
    # # frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_GRAY2RGB)

    frame_final = frame_processed

    faces_cv2 = face_cascade.detectMultiScale(frame_final, minSize=(int(0.1 * frame_final.shape[0]), int(0.1 * frame_final.shape[1])))
    primary_face_csv2_index, primary_face_csv2 = get_largest_shape([dict(x=x, y=y, width=w, height=h) for (x, y, w, h) in faces_cv2])
    faces_dlib = face_detector(frame_final, 1)
    primary_face_dlib_index, primary_face_dlib = get_largest_shape(
        [dict(x=rect.left(), y=rect.top(), width=rect.width(), height=rect.height()) for rect in faces_dlib])

    face_mtcnn, lms = fd_mtcnn.detect_face(frame_final)
    primary_face_mtcnn_index, primary_face_mtcnn = get_largest_shape(
        [dict(x=int(x0), y=int(y0), width=int(x1-x0), height=int(y1-y0)) for (y0, x1, y1, x0, score) in face_mtcnn])

    face = dict(
        img_width = img.shape[1], img_height = img.shape[0],
        bbox_dlib=[[primary_face_dlib['x'], primary_face_dlib['y']],
                      [primary_face_dlib['x'] + primary_face_dlib['width'],
                       primary_face_dlib['y'] + primary_face_dlib['height']]] if primary_face_dlib else [[None, None], [None, None]],
        bbox_opencv=[[primary_face_csv2['x'], primary_face_csv2['y']],
               [primary_face_csv2['x'] + primary_face_csv2['width'],
                primary_face_csv2['y'] + primary_face_csv2['height']]] if primary_face_csv2 else [[None, None], [None, None]],
        bbox_mtcnn=[[primary_face_mtcnn['x'], primary_face_mtcnn['y']],
                   [primary_face_mtcnn['x'] + primary_face_mtcnn['width'],
                    primary_face_mtcnn['y'] + primary_face_mtcnn['height']]] if primary_face_mtcnn else [[None, None],
                                                                                                      [None, None]],
        landmarks_dlib=[],
        landmarks_opencv=[],
        landmarks_mtcnn=[],
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
        face['landmarks_opencv'] = [[int(x[0]), int(x[1])] for x in landmarks2[0][0]]

    if len(lms) > 0:
        face['landmarks_mtcnn'] = [[lms[0][0], lms[5][0]], [lms[1][0], lms[6][0]], [lms[2][0], lms[7][0]], [lms[3][0], lms[8][0]], [lms[4][0], lms[9][0]]]

    face_landmarks = face['landmarks_dlib'] or face['landmarks_opencv']

    eye_imgs = []

    if len(face_landmarks) > 0:
        for (eye_name, eye_bb) in [('right_eye', cv2.boundingRect(np.array(face_landmarks[36: 41]))),
                                   ('left_eye', cv2.boundingRect(np.array(face_landmarks[42: 47])))]:
            eyebb_y1, eyebb_y2, eyebb_x1, eyebb_x2 = eye_bb[1] - int(0.5 * eye_bb[3]), \
                                                     eye_bb[1] + int(1.5 * eye_bb[3]), \
                                                     eye_bb[0] - int(0.5 * eye_bb[2]), \
                                                     eye_bb[0] + int(1.5 * eye_bb[2])

            eye1_frame = frame_final[eyebb_y1: eyebb_y2, eyebb_x1:eyebb_x2]
            # eye1_frame = cv2.cvtColor(eye1_frame, cv2.COLOR_GRAY2RGB)

            eye_imgs.append(eye1_frame)

            face[eye_name]['bbox'] = [[eyebb_x1, eyebb_y1], [eyebb_x2, eyebb_y2]]

            pupil = eye_landmark_detector.get_landmarks(eye1_frame)
            if pupil:
                face[eye_name]['pupil'] = [eyebb_x1 + round(pupil.pt[0]), eyebb_y1 + round(pupil.pt[1])]

    if len(eye_imgs) > 0:
        left_lms, right_lms = get_eye_landmarks(*eye_imgs, face['left_eye']['bbox'][0],  face['right_eye']['bbox'][0])
        face['left_eye']['gazeml'] = left_lms
        face['right_eye']['gazeml'] = right_lms

    return face

def generate_landmark_image(input_img, face):
    output_img = input_img.copy()
    for eye in ['left_eye', 'right_eye']:
        if 'bbox' in face[eye] and face[eye]['bbox'][0][0]:
            cv2.rectangle(output_img, pt1=tuple(face[eye]['bbox'][0]), pt2=tuple(face[eye]['bbox'][1]),
                      color=(255, 255, 0))

        pupil = face[eye].get('pupil')
        if pupil:
            cv2.circle(output_img, tuple(pupil), 3, (255, 0, 255), 1)

        if 'gazeml' in face[eye]:
            draw_pupil(output_img, face[eye]['gazeml'])

    if None not in face['bbox_dlib'][0]:
        cv2.rectangle(output_img, pt1= tuple(face['bbox_dlib'][0]), pt2=tuple(face['bbox_dlib'][1]), color=(0, 0, 255))
    for (x, y) in face['landmarks_dlib']:
        cv2.circle(output_img, (round(x), round(y)), 2, (0, 0, 255), -1)
    if None not in face['bbox_opencv'][0]:
        cv2.rectangle(output_img, pt1=tuple(face['bbox_opencv'][0]), pt2=tuple(face['bbox_opencv'][1]), color=(255, 0, 0))
    for (x, y) in face['landmarks_opencv']:
        cv2.circle(output_img, (round(x), round(y)), 2, (255, 0, 0), -1)
    if None not in face['bbox_mtcnn'][0]:
        cv2.rectangle(output_img, pt1=tuple(face['bbox_mtcnn'][0]), pt2=tuple(face['bbox_mtcnn'][1]), color=(0, 255, 0))
    for i, (x, y) in enumerate(face['landmarks_mtcnn']):
        cv2.circle(output_img, (round(x), round(y)), 2, (0, 255, 0), -1)

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

def get_eye_landmarks(left_eye_im, right_eye_im, left_pos, right_pos):
    inp_left = cv2.cvtColor(left_eye_im, cv2.COLOR_RGB2GRAY)
    inp_left = cv2.equalizeHist(inp_left)
    inp_left = cv2.resize(inp_left, (180, 108))[np.newaxis, ..., np.newaxis]

    inp_right = cv2.cvtColor(right_eye_im, cv2.COLOR_RGB2GRAY)
    inp_right = cv2.equalizeHist(inp_right)
    inp_right = cv2.resize(inp_right, (180, 108))[np.newaxis, ..., np.newaxis]

    input_array = np.concatenate([inp_left, inp_right], axis=0)
    pred_left, pred_right = elg_model.net.predict(input_array/255 * 2 - 1)

    left_landmarks = gazeml_landmarks(left_eye_im, pred_left, left_pos)
    right_landmarks = gazeml_landmarks(right_eye_im, pred_right, right_pos)

    return left_landmarks, right_landmarks

def gazeml_landmarks(im, pred, offset):
    lms = elg_model._calculate_landmarks(pred)
    lms = lms * (im.shape[1]*3/180.0, im.shape[0]*3/108.0 )
    lms = lms + offset
    gazeml_lms =  {
        'outerline': [],
        'innerline': [],
        'landmark_other':None,
        'landmarks_other': [],
        'pupilCenter': np.zeros((2,))
    }
    for i, lm in enumerate(np.squeeze(lms)):
        y, x = int(lm[0]), int(lm[1])
        if i < 8:
            gazeml_lms['outerline'].append([y, x])
        elif i < 16:
            gazeml_lms['innerline'].append([y, x])
            gazeml_lms['pupilCenter'] += (y,x)
        elif i < 17:
            gazeml_lms['landmark_other'] = [y, x]
        else:
            gazeml_lms['landmarks_other'].append([y, x])
    gazeml_lms['pupilCenter'] = (gazeml_lms['pupilCenter']/8).astype(np.int32)
    return gazeml_lms


def draw_pupil(im, lms):
    stroke = 2
    cv2.cv2.circle(im, tuple(lms['pupilCenter']), stroke, (255,255,0), -1)
    cv2.polylines(im, [np.array(lms['outerline']).reshape(-1,1,2)], isClosed=True, color=(125,255,125), thickness=stroke//2)
    cv2.polylines(im, [np.array(lms['innerline']).reshape(-1,1,2)], isClosed=True, color=(125,125,255), thickness=stroke//2)
    cv2.drawMarker(im, tuple(lms['landmark_other']), (255, 200, 200), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=stroke,
                          line_type=cv2.LINE_AA)
    for i, lm in enumerate(lms['landmarks_other']):
        cv2.drawMarker(im, tuple(lm), (255, 125, 125), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=stroke,
                              line_type=cv2.LINE_AA)
    return im
