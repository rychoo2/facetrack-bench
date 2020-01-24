import itertools
import os
import glob
import cv2
from libs.face import get_face, generate_landmark_image
from libs.utils import get_timestamp, get_datasets

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data"

def process_landmarks(input_path, output_path):
    header = True
    os.makedirs(output_path+'/images')
    with open('{}/landmarks.csv'.format(output_path), 'w') as fw:
        for f in glob.glob(input_path + "/images/*"):
            img, face = generate_landmark_for_file(f)
            basefilename = f.replace(input_path + "/", "")

            if header:
                row = create_landmarks_header()
                fw.write("{}\n".format(','.join(row)))
                header = False

            row = create_landmarks_row(face, basefilename)
            fw.write("{}\n".format(','.join([xstr(x) for x in row])))
            fw.flush()

            save_landmark_image(img, face, "{}/{}".format(output_path, basefilename))

def create_landmarks_row(face, basefilename):

    row_common = [basefilename, face['img_width'], face['img_height']]

    row = row_common.copy()
    row += face['bbox_dlib'][0]
    row += face['bbox_dlib'][1]
    row += face['bbox_opencv'][0]
    row += face['bbox_opencv'][1]
    for i in range(0, 68):
        landmark_x, landmark_y = face['landmarks_dlib'][i] if i < len(face['landmarks_dlib']) else [None, None]
        row += [landmark_x, landmark_y]
    for i in range(0, 68):
        landmark_x, landmark_y = face['landmarks_opencv'][i] if i < len(face['landmarks_opencv']) else [None, None]
        row += [landmark_x, landmark_y]
    for eye in ['left_eye', 'right_eye']:
        row += face[eye]['bbox'][0]
        row += face[eye]['bbox'][1]
    for eye in ['left_eye', 'right_eye']:
        pupil = face[eye].get('pupil')
        row += [pupil and pupil[0] or None, pupil and pupil[1] or None]

    return row

def create_landmarks_header():
    return list(itertools.chain.from_iterable([['filename', 'img_width', 'img_height'],
                                        ['face_dlib_x1', 'face_dlib_y1', 'face_dlib_x2', 'face_dlib_y2'],
                                        ['face_opencv_x1', 'face_opencv_y1', 'face_opencv_x2',
                                         'face_opencv_y2'],
                                        *[("landmark_dlib_{}_x".format(i), "landmark_dlib_{}_y".format(i)) for
                                          i in range(1, 69)],
                                        *[("landmark_opencv_{}_x".format(i), "landmark_opencv_{}_y".format(i))
                                          for i in range(1, 69)],
                                        ["left_eye_x1", "left_eye_y1", "left_eye_x2", "left_eye_y2"],
                                        ["right_eye_x1", "right_eye_y1", "right_eye_x2", "right_eye_y2"],
                                        ["left_pupil_x", "left_pupil_y", "right_pupil_x", "right_pupil_y"]]

                                       ))

def xstr(s):
    if s is None:
        return ''
    return str(s)

def generate_landmark_for_file(filepath):
    img = cv2.imread(filepath)
    face = get_face(img)
    return img, face

def save_landmark_image(input_img, face, output_path):
    output_img = generate_landmark_image(input_img, face)
    cv2.imwrite(output_path, output_img)

def generate_landmarks_for_datasets(input_root, output_root):
    path, datasets = get_datasets(input_root)
    for dataset in datasets:
        process_landmarks("{}/{}".format(path, dataset), "{}/{}".format(output_root, dataset))


if __name__ == '__main__':
    now = get_timestamp()
    output_path = "{}/landmarks/{}".format(train_data_dir, now)
    generate_landmarks_for_datasets(train_data_dir, output_path)

