import itertools
import os
import glob
from libs.utils import get_timestamp, get_datasets
import subprocess
import cv2
import shutil
import pandas as pd
from libs.utils import get_timestamp, get_latest_landmarks


train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data2"


def generate_features(raw_path, landmark_path, output_path):
    df = generate_output_df(raw_path, landmark_path)
    df['raw_image'] = df.apply(
        lambda row: os.path.join(os.path.dirname(os.path.relpath(raw_path)), row['image_path']),
        axis=1
    )

    df['landmark_image'] = df.apply(
        lambda row: os.path.join(os.path.dirname(os.path.relpath(landmark_path)), f"images/frame_{row['frame']}.jpg" ),
        axis=1
    )

    # features_calculation(df)
    include_target_features(df)

    output_df = df[['frame',
                    'raw_image', 'landmark_image', 'timestamp',
                                    # 'rel_face_x', 'rel_face_y', 'rel_face_size_x', 'rel_face_size_y',
                    'rel_target_x', 'rel_target_y',
                    'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
                    'gaze_angle_x','gaze_angle_y',
                    'pose_Tx', 'pose_Ty', 'pose_Tz',
                    'pose_Rx', 'pose_Ry', 'pose_Rz'
                ]]

    os.makedirs(output_path)
    output_df.to_csv("{}/features.csv".format(output_path), index=False)


def include_target_features(df):
    df['rel_target_x'] = df.x / df.screen_width
    df['rel_target_y'] = df.y / df.screen_height


def generate_output_df(input_csv, landmarks_csv):
    input_df = pd.read_csv(input_csv)
    landmarks_df = pd.read_csv(landmarks_csv)
    if input_df.index.size != landmarks_df.index.size:
        raise Exception("Input dataset length not matching with landmarks dataset")
    merged = input_df.merge(landmarks_df, on='frame', suffixes=['', '_2'])
    return merged



def generate_features_for_datasets(input_root, output_root):
    landmark_path, datasets = get_latest_landmarks(input_root)
    for dataset in datasets:
        generate_features("{}/raw/{}/positions.csv".format(input_root, dataset),
                          "{}/{}/landmarks.csv".format(landmark_path, dataset),
                          "{}/{}".format(output_root, dataset))


if __name__ == '__main__':
    now = get_timestamp()
    output_path = "{}/features/{}".format(train_data_dir, now)
    generate_features_for_datasets(train_data_dir, output_path)
