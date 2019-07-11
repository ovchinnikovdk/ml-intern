from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split

class StackModels(BaseEstimator, TransformerMixin):
    """Stacking models"""
    def __init__(self, models, x, y):
        if len(models) == 0:
            self.models = [RidgeClassifier(), SVC()]
        else:
            self.models = models[:]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)
        for model in self.models:
            model.fit(x_train, y_train)

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        x_new = x.copy()
        predicts = []
        for model in self.models:
            predicts.append(model.predict(x))
        for pred in predicts:
            x_new = np.append(x_new, pred.reshape(-1, 1), axis=1)
        return x
