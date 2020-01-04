import datetime
import os

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
    return os.listdir(path)
