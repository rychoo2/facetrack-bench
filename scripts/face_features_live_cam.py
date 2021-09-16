import sys
import os
import dlib
import glob
import cv2
import numpy as np
from imutils import face_utils
from libs.face import get_face

cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3

while True:
    ret_val, img = cam.read()

    faces = get_face(img)

    for face in faces:
        print(face)
        for (x, y) in face['landmarks']:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        for eye in ['left', 'right']:
            pupil = face['pupils'].get(eye)
            if pupil:
                cv2.circle(img, tuple(pupil), 1, (0, 255, 0), 2)

    cv2.imshow("Face landmarks", img)

    key = cv2.waitKey(1)
    if key > 0:
        break
cv2.destroyAllWindows()