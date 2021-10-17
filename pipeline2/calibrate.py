import tkinter as tk
from capturing_user_input_data import Capture
from step1_landmarks import run_openface_feature_extraction
from step2_features import generate_features

calibrate_path = '../train_data2/calibration'

N = 100

root = tk.Tk()
capture = Capture(root, N, calibrate_path)

raw_path = capture.output
dataset = raw_path.split("/")[-1]
root.mainloop()

landmarks_path = "{}/landmarks/{}".format(calibrate_path, dataset)
run_openface_feature_extraction(raw_path, landmarks_path)

raw_csv = "{}/positions.csv".format(raw_path)
landmarks_csv = "{}/landmarks.csv".format(landmarks_path)
features_path = "{}/features/{}".format(calibrate_path, dataset)
generate_features(raw_csv, landmarks_csv, features_path)