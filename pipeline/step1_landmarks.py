import itertools
import os
import glob
import cv2
from libs.face import get_faces
from libs.utils import get_timestamp

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data"

def process_landmarks(input_path, output_path):
    header = True
    os.makedirs(output_path+'/images')
    with open('{}/landmarks.csv'.format(output_path), 'w') as fw:
        for f in glob.glob(input_path + "/images/*"):
            img = cv2.imread(f)
            faces = get_faces(img)
            basefilename = f.replace(input_path + "/", "")

            if header:
                row = list(itertools.chain.from_iterable([['filename', 'img_width', 'img_height'],
                                                          ['face_x1', 'face_y1', 'face_x2', 'face_y2'],
                                                          *[("landmark{}_x".format(i), "landmark{}_y".format(i)) for i in range(1, 69)],
                                                          ["left_eye_x1", "left_eye_y1", "left_eye_x2", "left_eye_y2"],
                                                          ["right_eye_x1", "right_eye_y1", "right_eye_x2", "right_eye_y2"],
                                                          ["left_pupil_x", "left_pupil_y", "right_pupil_x", "right_pupil_y"]]

                ))
                fw.write("{}\n".format(','.join(row)))
                header = False

            row_common = [basefilename, str(img.shape[1]), str(img.shape[0])]
            if len(faces) == 0:
                fw.write("{}\n".format(','.join(row_common)))
                fw.flush()

            for face in faces:
                row = row_common.copy()
                row += face['bbox'][0]
                row += face['bbox'][1]
                for (x, y) in face['landmarks']:
                    cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
                    row += [x, y]
                for eye in ['left_eye', 'right_eye']:
                    row += face[eye]['bbox'][0]
                    row += face[eye]['bbox'][1]
                for eye in ['left_eye', 'right_eye']:
                    pupil = face[eye].get('pupil')
                    if pupil:
                        cv2.circle(img, tuple(pupil), 1, (0, 255, 0), 2)
                    row += [pupil and pupil[0] or '', pupil and pupil[1] or '']
                fw.write("{}\n".format(','.join([str(x) for x in row])))
                fw.flush()
            cv2.imwrite("{}/{}".format(output_path, basefilename), img)


def generate_landmarks_for_datasets(input_root, output_root):
    for dirname in os.listdir(input_root):
        process_landmarks("{}/{}".format(input_root, dirname), "{}/{}".format(output_root, dirname))

if __name__ == '__main__':
    raw_input_path = "{}/raw".format(train_data_dir)
    now = get_timestamp()
    output_path = "{}/landmarks/{}".format(train_data_dir, now)
    generate_landmarks_for_datasets(raw_input_path, output_path)

