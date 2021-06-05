import itertools
import os
import glob
from libs.utils import get_timestamp, get_datasets
import subprocess
import cv2
import shutil
import pandas as pd

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data2"
output_path = "{}/landmarks/{}".format(train_data_dir,  get_timestamp())
openface_bin_path = "/Users/juliusz/Sources/OpenFace/build/bin"


def run_openface_feature_extraction(input_path, output_path):
    openface_output_path = output_path + "/openface"
    subprocess.check_call([
        openface_bin_path +"/FeatureExtraction",
        "-verbose",
        "-fdir", input_path + "/images",
        "-out_dir", openface_output_path
    ])
    extract_images_from_video(openface_output_path + "/images.avi")
    shutil.move(openface_output_path+"/images", output_path)
    landmarks_csv = openface_output_path+"/images.csv"

    output_csv = merge_with_input(input_path +"/positions.csv", landmarks_csv)

    shutil.move(landmarks_csv, output_path+"/landmarks.csv")


def extract_images_from_video(videofile):
    output_path = os.path.dirname(videofile) + "/images"
    os.makedirs(output_path)
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(output_path + "/frame_%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count+=1


def merge_with_input(input_csv, landmarks_csv):
    input_df = pd.read_csv(input_csv)
    landmarks_df = pd.read_csv(landmarks_csv)



def generate_landmarks_for_datasets(input_root, output_root):
    path, datasets = get_datasets(input_root)
    for dataset in datasets:
        run_openface_feature_extraction("{}/{}".format(path, dataset), "{}/{}".format(output_root, dataset))


if __name__ == '__main__':

    generate_landmarks_for_datasets(train_data_dir, output_path)

