import sys
import os
import dlib
import glob
import cv2 as cv

predictor_path = "./data/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
cam = cv.VideoCapture(0)
color_green = (0,255,0)
line_width = 3

facemark = cv.face.createFacemarkLBF()
facemark.loadModel(cv.samples.findFile('../data/lbfmodel.yaml'))
cascade = cv.CascadeClassifier(cv.samples.findFile('../data/haarcascades/haarcascade_frontalface_alt2.xml'))

while True:
    ret_val, img = cam.read()
    frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    frame = cv.equalizeHist(frame)

    faces = cascade.detectMultiScale(frame, 1.05, 5, cv.CASCADE_SCALE_IMAGE, (100, 100))
    if len(faces) > 0:
        ok, landmarks = facemark.fit(frame, faces=faces)

        for marks in landmarks:
            cv.face.drawFacemarks(img, marks, color_green)

    cv.imshow("Image Landmarks", img)

    if cv.waitKey(1) == 27:
        break  # esc to quit
cv.destroyAllWindows()