import os
from libs.utils import get_timestamp, get_latest_landmarks
import pandas as pd

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data"

def generate_features(raw_path, landmark_path, output_path, features_calculation):
    raw_df = pd.read_csv(raw_path, names=['timestamp', 'gaze_x', 'gaze_y', 'screen_width', 'screen_height', 'filename'],
                         dtype={'timestamp': 'str'})
    landmark_df = pd.read_csv(landmark_path)
    df = landmark_df.set_index('filename').join(raw_df.set_index('filename'), on='filename')
    df['landmark_path'] = os.path.relpath(landmark_path)
    df['raw_path'] = os.path.relpath(raw_path)

    features_calculation(df)
    include_target_features(df)

    output_df = df[['raw_path', 'landmark_path', 'rel_target_x', 'rel_target_y', 'timestamp',
                                    'rel_face_x', 'rel_face_y', 'rel_face_size_x', 'rel_face_size_y',
                                    'rel_pose_x', 'rel_pose_y',
                                    'rel_left_eye_x', 'rel_left_eye_y', 'rel_right_eye_x', 'rel_right_eye_y',
                                    'rel_left_pupil_face_x', 'rel_left_pupil_face_y',
                                    'rel_eye_distance_x', 'rel_eye_distance_y', 'rel_right_pupil_face_x', 'rel_right_pupil_face_y',
                                    'rel_left_pupil_x', 'rel_left_pupil_y', 'rel_right_pupil_x', 'rel_right_pupil_y'
                                    ]]

    os.makedirs(output_path)
    output_df.to_csv("{}/features.csv".format(output_path))

def include_target_features(df):
    df['rel_target_x'] = df.gaze_x / df.screen_width
    df['rel_target_y'] = df.gaze_y / df.screen_height

def include_output_features_dlib(df):
    df['rel_face_x'] = df.face_dlib_x1 / df.img_width
    df['rel_face_y'] = df.face_dlib_y1 / df.img_height
    df['face_size_x'] = (df.face_dlib_x2 - df.face_dlib_x1)
    df['face_size_y'] = (df.face_dlib_y2 - df.face_dlib_y1)
    df['rel_face_size_x'] = df.face_size_x / df.img_width
    df['rel_face_size_y'] = df.face_size_y / df.img_height
    df['rel_pose_x'] = (df.landmark_dlib_34_x - df.face_dlib_x1) / df['face_size_x']
    df['rel_pose_y'] = (df.landmark_dlib_34_y - df.face_dlib_y1) / df['face_size_y']
    df['rel_left_eye_x'] = (df.left_eye_x1 - df.face_dlib_x1) / df['face_size_x']
    df['rel_left_eye_y'] = (df.left_eye_y1 - df.face_dlib_y1) / df['face_size_y']
    df['rel_right_eye_x'] = (df.right_eye_x1 - df.face_dlib_x1) / df['face_size_x']
    df['rel_right_eye_y'] = (df.right_eye_y1 - df.face_dlib_y1) / df['face_size_y']
    df['rel_left_pupil_face_x'] = (df.left_pupil_x - df.face_dlib_x1) / df['face_size_x']
    df['rel_left_pupil_face_y'] = (df.left_pupil_y - df.face_dlib_y1) / df['face_size_y']
    df['rel_right_pupil_face_x'] = (df.right_pupil_x - df.face_dlib_x1) / df['face_size_x']
    df['rel_right_pupil_face_y'] = (df.right_pupil_y - df.face_dlib_y1) / df['face_size_y']
    df['rel_eye_distance_x'] = (df.landmark_dlib_46_x - df.landmark_dlib_37_x) / df['face_size_x']
    df['rel_eye_distance_y'] = (df.landmark_dlib_46_y - df.landmark_dlib_37_y) / df['face_size_y']
    df['rel_left_pupil_x'] = (df.left_pupil_x - df.left_eye_x1) / (df.left_eye_x2 - df.left_eye_x1)
    df['rel_left_pupil_y'] = (df.left_pupil_y - df.left_eye_y1) / (df.left_eye_y2 - df.left_eye_y1)
    df['rel_right_pupil_x'] = (df.right_pupil_x - df.right_eye_x1) / (df.right_eye_x2 - df.right_eye_x1)
    df['rel_right_pupil_y'] = (df.right_pupil_y - df.right_eye_y1) / (df.right_eye_y2 - df.right_eye_y1)

def include_output_features_opencv(df):
    df['rel_face_x'] = df.face_opencv_x1 / df.img_width
    df['rel_face_y'] = df.face_opencv_y1 / df.img_height
    df['face_size_x'] = (df.face_opencv_x2 - df.face_opencv_x1)
    df['face_size_y'] = (df.face_opencv_y2 - df.face_opencv_y1)
    df['rel_face_size_x'] = df.face_size_x / df.img_width
    df['rel_face_size_y'] = df.face_size_y / df.img_height
    df['rel_pose_x'] = (df.landmark_opencv_34_x - df.face_opencv_x1) / df['face_size_x']
    df['rel_pose_y'] = (df.landmark_opencv_34_y - df.face_opencv_y1) / df['face_size_y']
    df['rel_left_eye_x'] = (df.left_eye_x1 - df.face_opencv_x1) / df['face_size_x']
    df['rel_left_eye_y'] = (df.left_eye_y1 - df.face_opencv_y1) / df['face_size_y']
    df['rel_right_eye_x'] = (df.right_eye_x1 - df.face_opencv_x1) / df['face_size_x']
    df['rel_right_eye_y'] = (df.right_eye_y1 - df.face_opencv_y1) / df['face_size_y']
    df['rel_left_pupil_face_x'] = (df.left_pupil_x - df.face_opencv_x1) / df['face_size_x']
    df['rel_left_pupil_face_y'] = (df.left_pupil_y - df.face_opencv_y1) / df['face_size_y']
    df['rel_right_pupil_face_x'] = (df.right_pupil_x - df.face_opencv_x1) / df['face_size_x']
    df['rel_right_pupil_face_y'] = (df.right_pupil_y - df.face_opencv_y1) / df['face_size_y']
    df['rel_eye_distance_x'] = (df.landmark_opencv_46_x - df.landmark_opencv_37_x) / df['face_size_x']
    df['rel_eye_distance_y'] = (df.landmark_opencv_46_y - df.landmark_opencv_37_y) / df['face_size_y']
    df['rel_left_pupil_x'] = (df.left_pupil_x - df.left_eye_x1) / (df.left_eye_x2 - df.left_eye_x1)
    df['rel_left_pupil_y'] = (df.left_pupil_y - df.left_eye_y1) / (df.left_eye_y2 - df.left_eye_y1)
    df['rel_right_pupil_x'] = (df.right_pupil_x - df.right_eye_x1) / (df.right_eye_x2 - df.right_eye_x1)
    df['rel_right_pupil_y'] = (df.right_pupil_y - df.right_eye_y1) / (df.right_eye_y2 - df.right_eye_y1)

def include_output_features_avg(df):
    df['rel_face_x'] = df[['face_opencv_x1', 'face_dlib_x1']].mean(axis=1) / df.img_width
    df['rel_face_y'] = df[['face_opencv_y1', 'face_dlib_y1']].mean(axis=1)/ df.img_height
    df['face_size_x'] = (df[['face_opencv_x2', 'face_dlib_x2']].mean(axis=1) - df[['face_opencv_x1', 'face_dlib_x1']].mean(axis=1))
    df['face_size_y'] = (df[['face_opencv_y2', 'face_dlib_y2']].mean(axis=1) - df[['face_opencv_y1', 'face_dlib_y1']].mean(axis=1))
    df['rel_face_size_x'] = df.face_size_x / df.img_width
    df['rel_face_size_y'] = df.face_size_y / df.img_height
    df['rel_pose_x'] = (df[['landmark_opencv_34_x', 'landmark_dlib_34_x']].mean(axis=1) - df[['face_opencv_x1', 'face_dlib_x1']].mean(axis=1)) / df['face_size_x']
    df['rel_pose_y'] = (df[['landmark_opencv_34_y', 'landmark_dlib_34_y']].mean(axis=1) - df[['face_opencv_y1', 'face_dlib_y1']].mean(axis=1)) / df['face_size_y']
    df['rel_left_eye_x'] = (df.left_eye_x1 - df[['face_opencv_x1', 'face_dlib_x1']].mean(axis=1)) / df['face_size_x']
    df['rel_left_eye_y'] = (df.left_eye_y1 - df[['face_opencv_y1', 'face_dlib_y1']].mean(axis=1)) / df['face_size_y']
    df['rel_right_eye_x'] = (df.right_eye_x1 - df[['face_opencv_x1', 'face_dlib_x1']].mean(axis=1)) / df['face_size_x']
    df['rel_right_eye_y'] = (df.right_eye_y1 - df[['face_opencv_y1', 'face_dlib_y1']].mean(axis=1)) / df['face_size_y']
    df['rel_left_pupil_face_x'] = (df.left_pupil_x - df[['face_opencv_x1', 'face_dlib_x1']].mean(axis=1)) / df['face_size_x']
    df['rel_left_pupil_face_y'] = (df.left_pupil_y - df[['face_opencv_y1', 'face_dlib_y1']].mean(axis=1)) / df['face_size_y']
    df['rel_right_pupil_face_x'] = (df.right_pupil_x - df[['face_opencv_x1', 'face_dlib_x1']].mean(axis=1)) / df['face_size_x']
    df['rel_right_pupil_face_y'] = (df.right_pupil_y - df[['face_opencv_y1', 'face_dlib_y1']].mean(axis=1)) / df['face_size_y']
    df['rel_eye_distance_x'] = (df[['landmark_opencv_46_x', 'landmark_dlib_46_x']].mean(axis=1) - df[['landmark_opencv_37_x', 'landmark_dlib_37_x']].mean(axis=1)) / df['face_size_x']
    df['rel_eye_distance_y'] = (df[['landmark_opencv_46_y', 'landmark_dlib_46_y']].mean(axis=1) - df[['landmark_opencv_37_y', 'landmark_dlib_37_y']].mean(axis=1)) / df['face_size_y']
    df['rel_left_pupil_x'] = (df.left_pupil_x - df.left_eye_x1) / (df.left_eye_x2 - df.left_eye_x1)
    df['rel_left_pupil_y'] = (df.left_pupil_y - df.left_eye_y1) / (df.left_eye_y2 - df.left_eye_y1)
    df['rel_right_pupil_x'] = (df.right_pupil_x - df.right_eye_x1) / (df.right_eye_x2 - df.right_eye_x1)
    df['rel_right_pupil_y'] = (df.right_pupil_y - df.right_eye_y1) / (df.right_eye_y2 - df.right_eye_y1)

def generate_features_for_datasets(input_root, output_root):
    landmark_path, datasets = get_latest_landmarks(input_root)
    for dataset in datasets:
        for (features_name, features_func) in [
            ('dlib', include_output_features_dlib),
            ('opencv', include_output_features_opencv),
            ('landmark_avg', include_output_features_avg),
        ]:
            generate_features("{}/raw/{}/positions.csv".format(input_root, dataset),
                              "{}/{}/landmarks.csv".format(landmark_path, dataset),
                              "{}/{}_{}".format(output_root, dataset, features_name),
                              features_calculation=features_func)

if __name__ == '__main__':
    now = get_timestamp()
    output_path = "{}/features/{}".format(train_data_dir, now)
    generate_features_for_datasets(train_data_dir, output_path)
