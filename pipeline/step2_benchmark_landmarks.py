import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

from libs.utils import get_latest_landmarks, get_timestamp


pd.options.display.float_format = "{:.4f}".format

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data"

def benchmark_landmarks_for_datasets(input_path, output_path):
    result = []
    os.makedirs(output_path)
    overall = pd.DataFrame()

    landmarks_path, datasets = get_latest_landmarks(input_path)

    for dataset in datasets:
        dataset_landmark_path = "{}/{}/landmarks.csv".format(landmarks_path, dataset)
        data = pd.read_csv(dataset_landmark_path)
        overall = overall.append(data)

        result += benchmark_landmarks(dataset, dataset_landmark_path, data)

    result += benchmark_landmarks('overall', landmarks_path, overall)

    output_df = pd.DataFrame(result)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(output_df.drop(output_df.columns[0], axis=1))
    output_df.to_csv(
        "{}/landmarks_benchmark.csv".format(output_path),
            header=['filename', 'dataset',
                    'metric', 'value'],
            index=False)

    return result


def benchmark_landmarks(dataset_name, filename, df):
    result = []
    all_count = len(df.index)
    metrics = {
        'face_dlib_detection': df['face_dlib_x1'].count(),
        'face_opencv_detection': df['face_opencv_x1'].count(),
        'face_any_detection': df[(df['face_dlib_x1'].notnull()) | (df['face_opencv_x1'].notnull())].shape[0],
        'landmark_dlib_detection': df['landmark_dlib_46_y'].count(),
        'landmark_opencv_detection': df['landmark_opencv_46_y'].count(),
        'landmark_any_detection':df[(df['landmark_dlib_46_y'].notnull()) | (df['landmark_opencv_46_y'].notnull())].shape[0],
        'eyes_detection': df[(df['left_eye_x1'].notnull()) & (df['right_eye_x1'].notnull())].shape[0],
        'pupils_detection':  df[(df['left_pupil_x'].notnull()) & (df['right_pupil_x'].notnull())].shape[0],
        'right_eye_detection': df['right_eye_x1'].count(),
        'left_eye_detection':  df['left_eye_x1'].count(),
        'right_pupil_detection':  df['right_pupil_x'].count(),
        'left_pupil_detection':  df['left_pupil_x'].count(),
    }
    result.append([os.path.relpath(filename), dataset_name, 'count', all_count])
    for name, value in metrics.items():
        result.append([os.path.relpath(filename), dataset_name, name, 1.0*value/all_count])
    return result

if __name__ == '__main__':
    now = get_timestamp()
    output_dir = "{}/landmarks_benchmark/{}".format(train_data_dir, now)
    benchmark_landmarks_for_datasets(train_data_dir, output_dir)
