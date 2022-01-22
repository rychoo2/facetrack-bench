import tkinter as tk
import pandas as pd
import os
from sklearn import ensemble
from pipeline2.models.sklearn_model_base import SklrearnModelBase
from pipeline2.step1_landmarks import run_live_openface_feature_extraction
from pipeline2.step2_features import generate_features_item
from pipeline2.calibrate import calibrate_path
from pipeline2.step3_machine_learning import training_columns, target_columns
from libs.utils import get_latest
from pipeline2.step2_features import generate_features
from libs.utils import get_timestamp

root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.state('zoomed')
root.attributes('-fullscreen', True)

canvas = tk.Canvas(root, bg='black', highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)
radius = 2.5
demo_on = True


def on_close(*args):
    global demo_on
    demo_on = False
    root.after(100, root.destroy)


root.bind('<Escape>', on_close)


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


def main():
    model = train_model()
    for prediction in predict_live(model):
        if not demo_on:
            return
        update_circle(prediction[0] * screen_width, prediction[1] * screen_height)
        root.update_idletasks()
        root.update()


def update_circle(x, y):
    canvas.delete("circle")
    draw_circle(x, y, radius=radius)


def draw_circle(x, y, radius=radius, color="red"):
    x0 = x - radius
    y0 = y - radius
    x1 = x + radius
    y1 = y + radius
    return canvas.create_oval(x0, y0, x1, y1, fill=color, outline=color, tags="circle")


if __name__ == "__main__":
    main()
