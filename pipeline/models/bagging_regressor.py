from pipeline.models.model_base import ModelBase

import sklearn.ensemble

class BaggingRegressor(ModelBase):

    def __init__(self):
        model = sklearn.ensemble.BaggingRegressor()
        self.model = model

    def train(self, input, target):
        self.model.fit(input, target)

    def predict(self, input):
        return self.model.predict(input)

