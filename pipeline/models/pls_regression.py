from pipeline.models.model_base import ModelBase

import sklearn.cross_decomposition

class PLSRegression(ModelBase):

    def __init__(self):
        model = sklearn.cross_decomposition.PLSRegression()
        self.model = model

    def train(self, input, target):
        self.model.fit(input, target)

    def predict(self, input):
        return self.model.predict(input)

