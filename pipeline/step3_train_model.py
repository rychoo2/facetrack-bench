from keras.models import Sequential
import pandas as pd
import numpy as np

from keras.layers import Dense
import keras
from keras import metrics

model = Sequential()
model.add(Dense(100, activation='linear', input_dim=2))
model.add(Dense(64, activation='linear'))
model.add(Dense(32, activation='linear'))
model.add(Dense(2, activation='linear'))

model.compile(optimizer='sgd',
              loss='mean_squared_error',
              metrics=[ metrics.mae])

# Generate dummy data
import numpy as np

data = pd.read_csv('../train_data/features/20191228150658055/headstill1/features.csv')

print(data)

# data = np.random.random((10, 2))
data = np.array([[0.1, 0.2],
                 [0.4, 0.3],
                 [0.5, 0.1],
                 [0.1, 0.2],
                 [0.4, 0.3],
                 [0.5, 0.1]
                 ])

# labels = np.random.randint(10, size=(1000, 2))
labels = np.array([[0.2, 0.4],
                  [0.8, 0.6],
                  [1, 0.2],
                   [0.2, 0.4],
                   [0.8, 0.6],
                   [1, 0.2]
                   ])
#
# print(labels)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=100)

output = model.predict(data)

print(np.array(list(zip(data,output))))

output2 = model.evaluate(data, labels)
print(output2)
