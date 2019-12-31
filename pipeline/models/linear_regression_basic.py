from pipeline.models.model_base import ModelBase
from sklearn.linear_model import LinearRegression

class LinearRegressionBasic(ModelBase):

    def __init__(self):
        model = LinearRegression()
        self.model = model

    def train(self, input, target):
        self.model.fit(input, target)

    def predict(self, input):
        return self.model.predict(input)

