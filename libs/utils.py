import datetime
import os

training_columns = ['rel_face_x', 'rel_face_y', 'rel_face_size_x', 'rel_face_size_y', 'rel_pose_x', 'rel_pose_y',
                      'rel_eye_distance_x', 'rel_eye_distance_y', 'rel_left_pupil_x', 'rel_left_pupil_y',
                      'rel_right_pupil_x', 'rel_right_pupil_y']

target_columns = ['rel_target_x', 'rel_target_y']

def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]


def get_latest_features(input_root):
    return get_latest(input_root, 'features')


def get_latest_landmarks(input_root):
    return get_latest(input_root, 'landmarks')


def get_latest(input_root, type):
    landmark_root_path = "{}/{}".format(input_root, type)
    last_generation = sorted(os.listdir(landmark_root_path), reverse=True)[0]
    landmark_input_path = "{}/{}".format(landmark_root_path, last_generation)
    return landmark_input_path, list_dirs(landmark_input_path)

def get_datasets(input_root):
    raw_input_path = "{}/raw".format(input_root)
    return raw_input_path, list_dirs(raw_input_path)

def list_dirs(path):
    return [x for x in os.listdir(path) if not '.DS_Store' in x]


