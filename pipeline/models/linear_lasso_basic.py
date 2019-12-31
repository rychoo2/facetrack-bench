from pipeline.models.model_base import ModelBase
from sklearn.linear_model import Lasso, LassoCV, MultiTaskLassoCV


class LinearLassoBasic(ModelBase):

    def __init__(self):
        model = MultiTaskLassoCV()
        self.model = model

    def train(self, input, target):
        self.model.fit(input, target)

    def predict(self, input):
        return self.model.predict(input)

