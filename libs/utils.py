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
    result = []
    for dirname in os.listdir(landmark_input_path):
        result.append((dirname, "{}/{}".format(landmark_input_path, dirname)))
    return result

def get_datasets(input_root):
    raw_input_path = "{}/raw".format(input_root)
    result = []
    for dirname in os.listdir(raw_input_path):
        result.append((dirname,  "{}/{}".format(raw_input_path, dirname)))
    return result
