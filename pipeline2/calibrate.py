from pipeline2.capturing_user_input_data import Capture
from pipeline2.step1_landmarks import run_openface_feature_extraction


calibrate_path = '../train_data2/calibration'


if __name__ == "__main__":
    N = 100

    capture = Capture(N, calibrate_path)

    raw_path = capture.output
    dataset = raw_path.split("/")[-1]
    capture.run()

    landmarks_path = "{}/landmarks/{}".format(calibrate_path, dataset)
    run_openface_feature_extraction(raw_path, landmarks_path)