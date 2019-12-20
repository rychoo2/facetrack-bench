import datetime
import itertools
import sys
import os
import dlib
import glob
import cv2
import numpy as np
from libs.face import get_faces
import ntpath

dataset = sys.argv[1] if len(sys.argv) > 1 else 'daylight_20191215123628'
input_path = "train_data/raw/{}/".format(dataset)
now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
output_path = "train_data/features/{}_{}".format(dataset, now)
header = True
os.makedirs(output_path+'/images')
fw = open('{}/features.csv'.format(output_path), 'w')

for f in glob.glob(input_path + "images/*"):
    img = cv2.imread(f)
    faces = get_faces(img)
    basefilename = f.replace(input_path, "")
    if header:
        row = list(itertools.chain.from_iterable([["filename"], *[("landmark{}_x".format(i), "landmark{}_y".format(i)) for i in range(1, 69)], ["left_pupil_x", "left_pupil_y"], ["right_pupil_x", "right_pupil_y"]]))
        header = False
    if len(faces) == 0:
        row = [basefilename]
        fw.write("{}\n".format(','.join(row)))
        fw.flush()
    for face in faces:
        row = [basefilename]

        for (x, y) in face['landmarks']:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
            row += [x, y]
        for eye in ['left', 'right']:
            pupil = face['pupils'].get(eye)
            if pupil:
                cv2.circle(img, tuple(pupil), 1, (0, 255, 0), 2)
            row += [pupil and pupil[0] or '', pupil and pupil[1] or '']
        fw.write("{}\n".format(','.join([str(x) for x in row])))
        fw.flush()
    cv2.imwrite("{}/{}".format(output_path, basefilename), img)
