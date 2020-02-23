from pipeline.models.model_base import ModelBase
import numpy as np

class CenterOfScreenModel(ModelBase):

    output_dimenstions = None
    def train(self, input, target):
        self.output_dimenstions = target.shape[1]
        pass

    def predict(self, input):
        return np.full((len(input), self.output_dimenstions), 0.5)
