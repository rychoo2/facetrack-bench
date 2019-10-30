import sys
import os
import dlib
import glob
import cv2
import numpy as np
from imutils import face_utils

predictor_path = "../data/shape_predictor_68_face_landmarks.dat"
eyes_cascade_path = '../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3

predictor = dlib.shape_predictor(predictor_path)

while True:
    ret_val, img = cam.read()

    frame_processed = img.copy()
    frame_processed = cv2.medianBlur(frame_processed, 5)
    frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)
    frame_processed = cv2.equalizeHist(frame_processed)

    frame_final = frame_processed #cv2.hconcat([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), frame_processed])


    dets = detector(frame_final, 1)
    print("Number of faces detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(frame_final, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))

        shape = face_utils.shape_to_np(shape)
        face_bb = cv2.boundingRect(shape)
        faceROI = frame_final[face_bb[1]: face_bb[1]+face_bb[3], face_bb[0]:face_bb[0] + face_bb[2]]
        # faceROI2 = frame_final[d.top():d.bottom(), d.left():d.right()]
        cv2.imshow("face", faceROI)
        # cv2.imshow("face2", faceROI2)

        for (eye_name, eye_bb) in [('eye_right', cv2.boundingRect(shape[36: 41])), ('eye_left', cv2.boundingRect(shape[42: 47]))]:
            eye1_frame = frame_final[eye_bb[1] -  int(0.8*eye_bb[3]): eye_bb[1]+int(1.8*eye_bb[3]), eye_bb[0] - int(0.8*eye_bb[2]):eye_bb[0] + int(1.8*eye_bb[2])]
            eye1_frame = cv2.pyrUp(eye1_frame)
            eye1_frame = cv2.equalizeHist(eye1_frame)
            cv2.imshow(eye_name, eye1_frame)

        for (x, y) in shape:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("Image", img)


    # for det in dets:
    #     cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
    # cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()