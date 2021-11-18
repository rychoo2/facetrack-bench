import cv2
import pandas as pd
import os
from sklearn import ensemble

from tkinter import Tk
from pipeline2.models.sklearn_model_base import SklrearnModelBase
from pynput.mouse import Controller

from pipeline2.step1_landmarks import run_live_openface_feature_extraction
from pipeline2.step2_features import generate_features_item
from pipeline2.calibrate import calibrate_path
from pipeline2.step3_machine_learning import training_columns, target_columns
from libs.utils import get_latest
from pipeline2.step2_features import generate_features
from libs.utils import get_timestamp

mouse = Controller()

root = Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()


def load_training_data():
    landmarks_input, _ = get_latest(calibrate_path, "landmarks")
    last_calibrate = landmarks_input.split('/')[-1]

    raw_csv = "{}/raw/{}/positions.csv".format(calibrate_path, last_calibrate)
    landmarks_csv = "{}/landmarks.csv".format(landmarks_input)
    features_path = "{}/features/{}/{}".format(calibrate_path, last_calibrate, get_timestamp())
    generate_features(raw_csv, landmarks_csv, features_path)

    df = pd.read_csv(os.path.join(features_path, "features.csv"))
    return df

def train_model():
    model = SklrearnModelBase(ensemble.BaggingRegressor)
    df = load_training_data()
    model.train(df[training_columns], df[target_columns])
    return model

def predict_live(model):
    for landmark_item in run_live_openface_feature_extraction("tmp"):
        features = generate_features_item(landmark_item, screen_width, screen_height)
        features_array = [features[key] for key in training_columns]
        prediction = model.predict([features_array])
        yield prediction[0]


def move_mouse(x, y):
    mouse.position = (x, y)

def main():
    print("loading model")
    model = train_model()

    for prediction in predict_live(model):

        move_mouse(prediction[0] * screen_width, prediction[1] * screen_height)
        if cv2.waitKey(1) == 27:
            break  # esc to quit


if __name__ == '__main__':
     main()