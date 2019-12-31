from pipeline.models.model_base import ModelBase

import sklearn.tree
import sklearn.multioutput


class DecisionTreeRegressor(ModelBase):

    def __init__(self):
        model = sklearn.tree.DecisionTreeRegressor()
        self.model = model

    def train(self, input, target):
        self.model.fit(input, target)

    def predict(self, input):
        return self.model.predict(input)
