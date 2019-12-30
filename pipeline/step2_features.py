import itertools
import os
import glob
import cv2
from libs.face import get_faces
from libs.utils import get_timestamp, get_last_landmarks
import pandas as pd

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data"

def generate_features(raw_path, landmark_path, output_path):
    raw_df = pd.read_csv(raw_path, names=['timestamp', 'gaze_x', 'gaze_y', 'screen_width', 'screen_height', 'filename'],
                         dtype={'timestamp': 'str'})
    landmark_df = pd.read_csv(landmark_path)
    df = landmark_df.set_index('filename').join(raw_df.set_index('filename'), on='filename')
    os.makedirs(output_path)
    df['landmark_path'] = os.path.relpath(landmark_path)
    df['raw_path'] = os.path.relpath(raw_path)
    df['rel_target_x'] = df.gaze_x/df.screen_width
    df['rel_target_y'] = df.gaze_y/df.screen_height
    df['rel_face_x'] = df.face_x1/df.img_width
    df['rel_face_y'] = df.face_y1/df.img_height
    df['face_size_x'] = (df.face_x2 - df.face_x1)
    df['face_size_y'] = (df.face_y2 - df.face_y1)
    df['rel_face_size_x'] = df.face_size_x/df.img_width
    df['rel_face_size_y'] = df.face_size_y/df.img_height
    df['rel_pose_x'] = (df.landmark34_x - df.face_x1)/df['face_size_x']
    df['rel_pose_y'] = (df.landmark34_y - df.face_y1)/df['face_size_y']
    df['rel_eye_distance_x'] = (df.landmark46_x - df.landmark37_x)/df['face_size_x']
    df['rel_eye_distance_y'] = (df.landmark46_y - df.landmark37_y)/df['face_size_y']
    df['rel_left_pupil_x'] = (df.left_pupil_x - df.left_eye_x1)/(df.left_eye_x2 - df.left_eye_x1)
    df['rel_left_pupil_y'] = (df.left_pupil_y - df.left_eye_y1)/(df.left_eye_y2 - df.left_eye_y1)
    df['rel_right_pupil_x'] = (df.right_pupil_x - df.right_eye_x1)/(df.right_eye_x2 - df.right_eye_x1)
    df['rel_right_pupil_y'] = (df.right_pupil_y - df.right_eye_y1)/(df.right_eye_y2 - df.right_eye_y1)

    df.to_csv("{}/features.csv".format(output_path), columns=['raw_path', 'landmark_path', 'rel_target_x', 'rel_target_y', 'timestamp',
                                    'rel_face_x', 'rel_face_y', 'rel_face_size_x', 'rel_face_size_y',
                                    'rel_pose_x', 'rel_pose_y', 'rel_eye_distance_x', 'rel_eye_distance_y',
                                    'rel_left_pupil_x', 'rel_left_pupil_y', 'rel_right_pupil_x', 'rel_right_pupil_y'
                                    ])

def generate_features_for_datasets(input_root, output_root):
    for (dirname, landmark_path) in get_last_landmarks(input_root):
        generate_features("{}/raw/{}/positions.csv".format(input_root, dirname),
                          "{}/landmarks.csv".format(landmark_path),
                          "{}/{}".format(output_root, dirname))

if __name__ == '__main__':
    now = get_timestamp()
    output_path = "{}/features/{}".format(train_data_dir, now)
    generate_features_for_datasets(train_data_dir, output_path)
