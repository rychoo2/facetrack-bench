import csv
import glob
import os
import time
from libs.utils import get_timestamp, get_datasets
import subprocess
import cv2
import shutil
import pandas as pd

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data2"
output_path = "{}/landmarks/{}".format(train_data_dir, get_timestamp())
openface_bin_path = os.environ.get('OPENFACE_BIN_PATH')


def run_live_openface_feature_extraction(output_path):
    openface_output_path = output_path + "/openface"
    shutil.rmtree(openface_output_path, ignore_errors=True)

    process = subprocess.Popen([
        openface_bin_path + "/FeatureExtraction",
        "-device", "0",
        "-oc", "H264",
        "-out_dir", openface_output_path
    ])

    csv_file_exists = False
    while not csv_file_exists:
        time.sleep(0.5)
        csv_list = glob.glob("{}/*csv".format(openface_output_path))
        if csv_list:
            csv_file = csv_list[0]
            csv_file_exists = True

    with open(csv_file, "r") as f:
        header_names = f.readline().split(",")

    frame = 0
    try:
        while True:
            landmarks_lines = read_file_last_nlines(csv_file, 2)
            if len(landmarks_lines[1]) == len(header_names):
                landmarks = landmarks_lines[1]
            else:
                landmarks = landmarks_lines[0]
                print("last line not full")
            last_frame = int(landmarks[0])
            if last_frame > frame:
                landmarks_dict = dict(zip(header_names, landmarks))
                frame = last_frame
                yield landmarks_dict

    except GeneratorExit:
        process.terminate()
        process.wait()
        print("finished")
        raise


def run_openface_feature_extraction(input_path, output_path):
    openface_output_path = output_path + "/openface"

    subprocess.check_call([
        openface_bin_path + "/FeatureExtraction",
        "-verbose",
        "-fdir", input_path + "/images",
        "-oc H264",
        "-out_dir", openface_output_path
    ])
    extract_images_from_video(openface_output_path + "/images.avi")
    shutil.move(openface_output_path + "/images", output_path)
    landmarks_csv = openface_output_path + "/images.csv"

    df = pd.read_csv(landmarks_csv)

    df.insert(1, 'landmark_image', df.apply(
        lambda row: image_filename(row['frame']),
        axis=1)
              )

    df.to_csv(output_path + "/landmarks.csv", index=False)


def extract_images_from_video(videofile):
    output_path = os.path.dirname(videofile)
    os.makedirs(output_path + "/images")
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    count = 1
    while success:
        cv2.imwrite(os.path.join(output_path, image_filename(count)), image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1


def read_file_last_nlines(file, nlines):
    with open(file,"rb") as f:
        f.seek(0, os.SEEK_END)
        endf = position = f.tell()
        linecnt = 0
        while position >= 0:
            f.seek(position)
            next_char = f.read(1)
            if next_char == b"\n" and position != endf - 1:
                linecnt += 1
            if linecnt == nlines:
                break
            position -= 1

        if position < 0:
            f.seek(0)

        return [line.rstrip().split(',') for line in f.read().decode().split('\n')]


def image_filename(frame):
    return f"images/frame_{int(frame):d}.jpg"


def generate_landmarks_for_datasets(input_root, output_root):
    path, datasets = get_datasets(input_root)
    for dataset in datasets:
        run_openface_feature_extraction("{}/{}".format(path, dataset), "{}/{}".format(output_root, dataset))


if __name__ == '__main__':
    generate_landmarks_for_datasets(train_data_dir, output_path)
