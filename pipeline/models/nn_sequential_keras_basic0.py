from pipeline.models.model_base import ModelBase
from keras.models import Sequential

from keras.layers import Dense
class NNSequentialKerasBasic0(ModelBase):

    def __init__(self):
        model = Sequential()
        model.add(Dense(12, activation='linear', input_dim=12))
        model.add(Dense(2, activation='linear'))

        model.compile(optimizer='sgd',
                      loss='mean_squared_error')

        self.model = model


    def train(self, input, target):
        self.model.fit(input, target, epochs=100)

    def predict(self, input):
        return self.model.predict(input)

