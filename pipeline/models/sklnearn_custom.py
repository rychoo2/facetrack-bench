from pipeline.models.model_base import ModelBase

import sklearn.linear_model

class SklearnCustom(ModelBase):

    def __init__(self):
        model = sklearn.linear_model.ARDRegression()
        self.model = model

    def train(self, input, target):
        self.model.fit(input, target)

    def predict(self, input):
        return self.model.predict(input)

