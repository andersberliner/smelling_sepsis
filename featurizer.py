# featurizer.py
# Anders Berliner
# 20160106
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd

class PolynomialFeaturizer(object):
    def __init__(self, n=4):
        self.n = n

    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X):
        # X is test or train Data => a df
        #   index  data
        # col 0 is time (minutes)
        # col 1:p are time course data
        # for each col, each trial of X
        # regress t^0, t^1, ..., t^n, col
        # return pd_series with same indices
        # values are regression coefficients
        poly = PolynomialFeatures(degree=n)
        t = poly.fit_transform(x[:,0])
        model = LinearRegression(fit_intercept=False)


        # self.coef_ = self.coef_.apply(lambda x: self._regress(x, t, model, number_of_spots))


    def _regress(self, X, t, model, number_of_spots=219):
        number_of_spots = X.iloc[0].shape[1]-1
        coef_ = X.copy()
        scores_ = X.apply(lambda x: np.zeros(number_of_spots))
        for trial_index, x in enumerate(X):
            # regress coefficients are (poly order +1 )x(n_spots)
            coefficients = np.zeros(self.n+1, number_of_spots)
            scores = np.zeros(number_of_spots)
            for column_index in x.shape(1):
                spot_index = column_index - 1
                if column_index == 0:
                    pass
                else:
                    model.fit(t, x[:,column_index])
                    coefficients[:, spot_index] = model.coef_
                    scores[spot_index] = model.score(t, x[:, column_index])
            coef_.iloc[trial_index] = coefficients
            scores_.iloc[trial_index] = scores
        return coef_, scores_



class KineticsFeaturizer(object):
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X):
        pass

class LongitudinalFeaturizer(object):
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X):
        pass
