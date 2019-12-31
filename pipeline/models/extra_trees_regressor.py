from pipeline.models.model_base import ModelBase

import sklearn.ensemble

class ExtraTreesRegressor(ModelBase):

    def __init__(self):
        model = sklearn.ensemble.ExtraTreesRegressor()
        self.model = model

    def train(self, input, target):
        self.model.fit(input, target)

    def predict(self, input):
        return self.model.predict(input)

