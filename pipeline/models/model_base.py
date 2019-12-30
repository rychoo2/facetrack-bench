from abc import abstractmethod, ABC
from sklearn.metrics import mean_absolute_error


class ModelBase(ABC):

    def train(self, input, target):
        pass

    def predict(self, input):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    @staticmethod
    def evaluate(predicted, target):
        return mean_absolute_error(target, predicted)

    @property
    def name(self):
        return self.__class__.__name__
