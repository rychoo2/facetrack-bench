from pipeline.models.model_base import ModelBase

import sklearn.neural_network


class MLPRegressor(ModelBase):

    def __init__(self):
        model = sklearn.neural_network.MLPRegressor()
        self.model = model

    def train(self, input, target):
        self.model.fit(input, target)

    def predict(self, input):
        return self.model.predict(input)
