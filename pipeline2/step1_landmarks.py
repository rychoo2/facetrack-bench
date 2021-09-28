import os
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

    df = pd.read_csv(landmarks_csv)

    df.insert(1, 'landmark_image', df.apply(
        lambda row: image_filename(row['frame']),
        axis=1)
    )

    df.to_csv(output_path+"/landmarks.csv", index=False)


def extract_images_from_video(videofile):
    output_path = os.path.dirname(videofile)
    os.makedirs(output_path + "/images")
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    count = 1
    while success:
        cv2.imwrite(os.path.join(output_path, image_filename(count)), image)  # save frame as JPEG file
        success, image = vidcap.read()
        count+=1

def image_filename(frame):
    return f"images/frame_{int(frame):d}.jpg"

def generate_landmarks_for_datasets(input_root, output_root):
    path, datasets = get_datasets(input_root)
    for dataset in datasets:
        run_openface_feature_extraction("{}/{}".format(path, dataset), "{}/{}".format(output_root, dataset))


if __name__ == '__main__':

    generate_landmarks_for_datasets(train_data_dir, output_path)

