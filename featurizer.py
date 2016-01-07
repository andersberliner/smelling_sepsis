# featurizer.py
# Anders Berliner
# 20160106
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd

class PolynomialFeaturizer(object):
    def __init__(self, n=4, reference_time=0):
        self.n = n
        self.reference_time = reference_time

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
        ## NEED to incorporate reference_time
        self.X = X
        poly = PolynomialFeatures(degree=self.n)
        model = LinearRegression(fit_intercept=False)

        self.coef_, self._scores_ = self._regress(X, model, poly)
        return self.coef_

    def score(self, X=None):
        # ugly, but trying to match sklearn form
        return self.scores_

    def predict(self):
        # first column of Z is time
        # we will replace the other columns with regressed data
        Z = X.copy()
        poly = PolynomialFeatures(degree=self.n)
        for trial_index, (coefficients, x) in enumerate(izip(self.coef_, self.X)):
            # reshape required by t
            t = poly.fit_transform(x[:,0].reshape(-1,1))
            z = np.zeros(x.shape)
            # first column is time
            z[:,0] = x[:,0]
            # columns up to reference time are just 0 and were not regressed
            z[:reference_time, 1:] = 0
            # columns after reference_time were regressed with coefficients
            z[reference_time:, 1:] = np.dot(t, coefficients)
            Z.iloc[trial_index] = z
        return Z


    def _regress(self, X, model, poly):
        number_of_spots = X.iloc[0].shape[1]-1
        self.number_of_spots = number_of_spots
        coef_ = X.copy()
        scores_ = X.apply(lambda x: np.zeros(number_of_spots))
        for trial_index, x in enumerate(X):
            # regress coefficients are (poly order +1 )x(n_spots)
            coefficients = np.zeros(((self.n+1), number_of_spots))
            scores = np.zeros(number_of_spots)
            # number of times different for each observation
            # QUESTION - what for trials of different lengths?
            #   -> maybe deal with this in the fit/predict steps?
            t = poly.fit_transform(x[:,0])

            for column_index in x.shape[1]:
                spot_index = column_index - 1
                if column_index == 0:
                    pass
                else:
                    # only fit data past the reference_time
                    # other data is 0 from preprocessing
                    model.fit(t, x[self.reference_time:,column_index])
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
