from pipeline.models.model_base import ModelBase
from keras.models import Sequential

from keras.layers import Dense
class NNSequentialKerasBasic0(ModelBase):

    def __init__(self):
        self.model = None


    def train(self, input, target):
        model = Sequential()
        model.add(Dense(12, activation='linear', input_dim=input.shape[1]))
        model.add(Dense(target.shape[1], activation='linear'))

        model.compile(optimizer='sgd',
                      loss='mean_squared_error')

        self.model = model
        self.model.fit(input, target, epochs=100)

    def predict(self, input):
        return self.model.predict(input)

