from abc import abstractmethod, ABC
from sklearn.metrics import mean_absolute_error

from pipeline.models import ModelBase
import numpy as np

class SklrearnModelBase(ModelBase):

    def __init__(self, modelClass):
       self.modelClass = modelClass
       self.models = None

    def _create_model(self):
        return self.modelClass()

    def train(self, input, target):
        self.models = [self._create_model() for i in range(0, target.shape[1])]
        target_ndarray = target.to_numpy()
        for i in range(0, target.shape[1]):
            self.models[i].fit(input, target_ndarray[:,i])

    def predict(self, input):
        predictions = [self.__normalize_dims(m.predict(input)) for m in self.models]
        return np.concatenate(predictions, axis=1)

    def save(self, path):
        pass

    def load(self, path):
        pass

    @staticmethod
    def __normalize_dims(ndarray):
        if len(np.shape(ndarray))>1:
            return ndarray
        else:
            return np.expand_dims(ndarray, axis=1)

    @property
    def name(self):
        return 'Sklearn_'+self.modelClass.__name__
