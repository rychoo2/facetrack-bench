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
        target_ndarray = target.to_numpy()
        evaluations = [mean_absolute_error(target_ndarray, predicted)]
        if target.shape[1] > 1:
            for i in range(0, target.shape[1]):
                evaluations.append(mean_absolute_error(target_ndarray[:,i], predicted[:,i]))
        return evaluations

    @property
    def name(self):
        return self.__class__.__name__
