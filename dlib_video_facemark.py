import sys
import os
import dlib
import glob
import cv2
import numpy as np
from imutils import face_utils

predictor_path = "./data/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3

predictor = dlib.shape_predictor(predictor_path)

while True:
    ret_val, img = cam.read()

    frame_processed = img.copy()
    # frame_processed = cv2.medianBlur(frame_processed, 5)
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
        # cv2.polylines(img, [shape], True, color_green, 4, 4)

        for (x, y) in shape:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("Image", img)
    # for det in dets:
    #     cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
    # cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()