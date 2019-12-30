from keras.models import Sequential
import pandas as pd
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from keras.layers import Dense
import keras
from keras import metrics

from pipeline.models.center_of_screen_model import CenterOfScreenModel
from pipeline.models import ModelBase
from pipeline.models import NNSequentialKeras

model = Sequential()
model.add(Dense(12, activation='linear', input_dim=12))
model.add(Dense(2, activation='linear'))

model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=[ metrics.mae])

models = [CenterOfScreenModel(), NNSequentialKeras()]

data = pd.read_csv('../train_data/features/20191230104532257/headstill1/features.csv')
data.dropna(subset=['rel_target_x','rel_target_y'], inplace=True)
# data.dropna(inplace=True)
data.fillna(0.5, inplace=True)
input = data[['rel_face_x','rel_face_y','rel_face_size_x','rel_face_size_y','rel_pose_x','rel_pose_y',
              'rel_eye_distance_x','rel_eye_distance_y','rel_left_pupil_x','rel_left_pupil_y','rel_right_pupil_x',
              'rel_right_pupil_y']]
target = data[['rel_target_x', 'rel_target_y']]

result = []

for model in models:
    model.train(input, target)
    output = model.predict(input)
    benchmark = model.evaluate(output, target)
    result.append([model.name, 'headstill1', benchmark])

output = model.predict(input)
pp.pprint(result)

# print(input)
# print(output)
# print(target)
# # print(np.array(list(zip(output, target))))
