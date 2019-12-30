from keras.models import Sequential
import pandas as pd
import numpy as np
import pprint
import os

from libs.utils import get_latest_features

pp = pprint.PrettyPrinter(indent=4)

from keras.layers import Dense
import keras
from keras import metrics

from pipeline.models import CenterOfScreenModel,  NNSequentialKerasBasic,  NNSequentialKerasBasic0

train_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/../train_data"


model = Sequential()
model.add(Dense(12, activation='linear', input_dim=12))
model.add(Dense(2, activation='linear'))

model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=[ metrics.mae])

models = [CenterOfScreenModel(), NNSequentialKerasBasic(), NNSequentialKerasBasic0()]

result = []

overall_input = pd.DataFrame()
overall_target = pd.DataFrame()

for dataset, path in get_latest_features(train_data_dir):
    data = pd.read_csv("{}/features.csv".format(path))
    data.dropna(subset=['rel_target_x', 'rel_target_y'], inplace=True)
    data.fillna(0.5, inplace=True)

    input = data[['rel_face_x', 'rel_face_y', 'rel_face_size_x', 'rel_face_size_y', 'rel_pose_x', 'rel_pose_y',
                  'rel_eye_distance_x', 'rel_eye_distance_y', 'rel_left_pupil_x', 'rel_left_pupil_y',
                  'rel_right_pupil_x',
                  'rel_right_pupil_y']]
    target = data[['rel_target_x', 'rel_target_y']]

    overall_input = overall_input.append(input)
    overall_target = overall_target.append(target)

    for model in models:
        model.train(input, target)
        output = model.predict(input)
        benchmark = model.evaluate(output, target)
        result.append([model.name, dataset, benchmark])

for model in models:
    model.train(overall_input, overall_target)
    output = model.predict(overall_input)
    benchmark = model.evaluate(output, overall_target)
    result.append([model.name, 'overall', benchmark])

pp.pprint(result)

# print(input)
# print(output)
# print(target)
# # print(np.array(list(zip(output, target))))
