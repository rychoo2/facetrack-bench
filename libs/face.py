import dlib
import cv2
from imutils import face_utils
from libs.eye_landmark import detect_pupil
import os

curdir = os.path.dirname(os.path.realpath(__file__))
predictor_path = curdir + "/../../data/shape_predictor_68_face_landmarks.dat"
eyes_cascade_path = curdir + '/../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
face_detector = dlib.get_frontal_face_detector()
landmarks_detector = dlib.shape_predictor(predictor_path)

def get_faces(img):
    faces = []
    frame_processed = img.copy()
    frame_processed = cv2.medianBlur(frame_processed, 5)
    frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)
    frame_processed = cv2.equalizeHist(frame_processed)

    frame_final = frame_processed

    dets = face_detector(frame_final, 1)
    for k, d in enumerate(dets):
        _shape = landmarks_detector(frame_final, d)
        shape = face_utils.shape_to_np(_shape)

        face = dict(bbox = [[ d.left(), d.top()],[d.right(), d.bottom()]], landmarks = shape.tolist(), left_eye=dict(), right_eye=dict())

        for (eye_name, eye_bb) in [('right_eye', cv2.boundingRect(shape[36: 41])), ('left_eye', cv2.boundingRect(shape[42: 47]))]:
            eyebb_y1, eyebb_y2, eyebb_x1, eyebb_x2 = eye_bb[1] - int(0.8*eye_bb[3]),\
                eye_bb[1] + int(1.8 * eye_bb[3]), \
                eye_bb[0] - int(0.8 * eye_bb[2]), \
                eye_bb[0] + int(1.8 * eye_bb[2])

            eye1_frame = frame_final[eyebb_y1: eyebb_y2, eyebb_x1:eyebb_x2]
            eye1_frame = cv2.cvtColor(eye1_frame, cv2.COLOR_GRAY2RGB)
            pupil = detect_pupil(eye1_frame)

            face[eye_name]['bbox'] = [[eyebb_x1, eyebb_y1], [eyebb_x2, eyebb_y2]]
            if pupil:
                face[eye_name]['pupil'] = [eyebb_x1 + round(pupil.pt[0]), eyebb_y1 + round(pupil.pt[1])]
        faces.append(face)
    return faces
