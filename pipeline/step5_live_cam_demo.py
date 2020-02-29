import sys
import os
import dlib
import glob
import cv2
import numpy as np
import pandas as pd
from pynput import mouse
import datetime

from libs.face import get_face, generate_landmark_image
from libs.utils import get_timestamp, target_columns, training_columns
from tkinter import Tk
from pipeline.models.extra_trees_regressor import ExtraTreesRegressor
from pipeline.step1_landmarks import create_landmarks_row, create_landmarks_header
from pipeline.step3_features import include_output_features_dlib
from pynput.mouse import Button, Controller
mouse = Controller()

root = Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()


cam = cv2.VideoCapture(0)

training_features_path = os.path.dirname(os.path.realpath(__file__)) + "/../train_data/features/20200124181140939/20200124175712535_opencv/features.csv"
features_func = include_output_features_dlib

def load_training_data():
    df = pd.read_csv(training_features_path)
    df.dropna(subset=target_columns, inplace=True)
    df.fillna(0.5, inplace=True)
    return df

def train_model():
    model = ExtraTreesRegressor()
    df = load_training_data()
    model.train(df[training_columns], df[target_columns])
    return model

def create_landmarks(face):
    return pd.DataFrame([create_landmarks_row(face, "")], columns=create_landmarks_header(), dtype='float64')

def generate_features(img):
    face = get_face(img)
    output_img = generate_landmark_image(img, face)
    landmarks = create_landmarks(face)
    include_output_features_dlib(landmarks)
    return landmarks[training_columns], output_img

def move_mouse(x, y):
    mouse.position = (x, y)

def main():
    print("loading model")
    model = train_model()
    while True:
        ret_val, img = cam.read()
        features, output_img = generate_features(img)

        features.dropna(inplace=True)
        if features.shape[0] > 0:
            prediction = model.predict(features)[0]
            print(prediction)
            # move_mouse(prediction[0] * screen_width, prediction[1] * screen_height)
            cv2.circle(output_img, (int(prediction[0] * img.shape[1]), int(prediction[0] * img.shape[0])), radius=20, color=(255, 255, 255))

            print(features.iloc[0])
            features_debug_x =  ",".join(["{0:.2f}".format(x) for x in features[[c for c in features.columns if c.endswith('x')]].iloc[0]])
            features_debug_y =  ",".join(["{0:.2f}".format(x) for x in features[[c for c in features.columns if c.endswith('y')]].iloc[0]])

            print(features_debug_x)
            print(features_debug_y)
            cv2.putText(output_img,features_debug_x, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, color=(0, 0, 0))
            cv2.putText(output_img,features_debug_y, (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, color=(0, 0, 0))

            print((int(prediction[0] * img.shape[1]), int(prediction[0] * img.shape[0])))
        else:
            print(None)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        cv2.imshow("output", output_img)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()