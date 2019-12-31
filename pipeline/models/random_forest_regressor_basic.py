from pipeline.models.model_base import ModelBase

import sklearn.ensemble

class RandomForestRegressorBasic(ModelBase):

    def __init__(self):
        model = sklearn.ensemble.RandomForestRegressor()
        self.model = model

    def train(self, input, target):
        self.model.fit(input, target)

    def predict(self, input):
        return self.model.predict(input)

