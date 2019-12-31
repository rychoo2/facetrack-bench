from pipeline.models.model_base import ModelBase

import sklearn.tree
import sklearn.multioutput


class SklearnCustom(ModelBase):

    def __init__(self):
        model = sklearn.tree.ExtraTreeRegressor()
        self.model = model

    def train(self, input, target):
        self.model.fit(input, target)

    def predict(self, input):
        return self.model.predict(input)
