from pipeline.models.model_base import ModelBase

import sklearn.linear_model


class MultiTaskLassoCV(ModelBase):

    def __init__(self):
        model = sklearn.linear_model.MultiTaskLassoCV(max_iter=10000, tol=0.00001)
        self.model = model

    def train(self, input, target):
        self.model.fit(input, target)

    def predict(self, input):
        return self.model.predict(input)
