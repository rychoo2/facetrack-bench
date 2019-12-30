from pipeline.models.model_base import ModelBase
import numpy as np

class CenterOfScreenModel(ModelBase):

    def train(self, input, target):
        pass

    def predict(self, input):
        return np.full((len(input), 2), 0.5)
