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
                row = list(itertools.chain.from_iterable([['filename', 'img_width', 'img_height'],
                                                          ['face_x1', 'face_y1', 'face_x2', 'face_y2'],
                                                          ['face_alt_x1', 'face_alt_y1', 'face_alt_x2', 'face_alt_y2'],
                                                          *[("landmark{}_x".format(i), "landmark{}_y".format(i)) for i in range(1, 69)],
                                                          ["left_eye_x1", "left_eye_y1", "left_eye_x2", "left_eye_y2"],
                                                          ["right_eye_x1", "right_eye_y1", "right_eye_x2", "right_eye_y2"],
                                                          ["left_pupil_x", "left_pupil_y", "right_pupil_x", "right_pupil_y"]]

                ))
                fw.write("{}\n".format(','.join(row)))
                header = False

            row_common = [basefilename, str(img.shape[1]), str(img.shape[0])]

            row = row_common.copy()
            row += face['bbox'][0]
            row += face['bbox'][1]
            row += face['bbox2'][0]
            row += face['bbox2'][1]
            for i in range(0,68):
                landmark_x, landmark_y = face['landmarks'][i] if i < len(face['landmarks']) else [None, None]
                row += [landmark_x, landmark_y]
            for eye in ['left_eye', 'right_eye']:
                row += face[eye]['bbox'][0]
                row += face[eye]['bbox'][1]
            for eye in ['left_eye', 'right_eye']:
                pupil = face[eye].get('pupil')
                row += [pupil and pupil[0] or '', pupil and pupil[1] or '']
            fw.write("{}\n".format(','.join([xstr(x) for x in row])))
            fw.flush()

            save_landmark_image(img, face, "{}/{}".format(output_path, basefilename))

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

