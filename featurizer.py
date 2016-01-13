# featurizer.py
# Anders Berliner
# 20160106
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from itertools import izip
import time
from output_capstone import print_to_file_and_terminal as ptf

class PolynomialFeaturizer(object):
    def __init__(self, n=4, reference_time=0,
            verbose=True, gridsearch=False, logfile=None,
            regressor_type='OLS', n_jobs=1):
        self.n = n
        self.reference_time = reference_time
        self.verbose = verbose
        self.logfile = logfile
        self.gridsearch = gridsearch
        self.n_jobs = n_jobs

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
        # avoid to save on space self.X = X
        poly = PolynomialFeatures(degree=self.n)

        bestmodel = []
        best_loss = None
        if self.gridsearch:
            # can use Ridge or Lasso here => gridsearch
            # random_forest_grid = {'max_depth': [3, None],
            #           'max_features': ['sqrt', 'log2', None],
            #           'min_samples_split': [1, 2, 4],
            #           'min_samples_leaf': [1, 2, 4],
            #           'bootstrap': [True, False],
            #           'n_estimators': [10, 20, 40],
            #           'random_state': [1]}
            # rf_gridsearch = GridSearchCV(RandomForestRegressor(),
            #                  random_forest_grid,
            #                  n_jobs=-1,
            #                  verbose=True,
            #                  scoring='mean_squared_error')
            # rf_gridsearch.fit(X_train, y_train)
            model = LassoCV(fit_intercept=False, n_jobs=self.n_jobs)
            self.regressor_type = 'LassoCV'
        else:
            model = LinearRegression(fit_intercept=False, n_jobs=self.n_jobs)

        coef_, scores_ = self._regress(X, model, poly)
        return coef_, scores_

    # def score(self, X=None):
    #     # ugly, but trying to match sklearn form
    #     # better to modify so it's using a predict, score method
    #     # and calculating residual
    #     return self.scores_

    # need to change this to accept the input array
    # DONE
    def predict(self, Z, coefs):
        # first column of Z is time
        # we will replace the other columns with regressed data

        # clean-up from before
        # Z = self.X.copy()

        poly = PolynomialFeatures(degree=self.n)
        for trial_index, (coefficients, x) in enumerate(izip(coefs, Z)):
            # reshape required by t
            t = poly.fit_transform((x[:,0]).reshape(-1,1))
            # only regress on data past reference time
            t = t[self.reference_time:]

            z = np.zeros(x.shape)
            # first column is time
            z[:,0] = x[:,0]
            # columns up to reference time are just 0 and were not regressed
            z[:self.reference_time, 1:] = 0
            # columns after reference_time were regressed with coefficients
            z[self.reference_time:, 1:] = np.dot(t, coefficients)
            Z.iloc[trial_index] = z
        return Z


    def _regress(self, X, model, poly):
        # timing of the regress step
        start = time.time()

        number_of_spots = X.iloc[0].shape[1]-1
        self.number_of_spots = number_of_spots
        coef_ = X.copy()
        scores_ = X.apply(lambda x: np.zeros(number_of_spots))
        for trial_index, x in enumerate(X):
            if trial_index % 100 == 0:
                if self.verbose:
                    ptf( 'Featurizing trial %d'%  trial_index, self.logfile)
            # regress coefficients are (poly order +1 )x(n_spots)
            coefficients = np.zeros(((self.n+1), number_of_spots))
            scores = np.zeros(number_of_spots)
            # number of times different for each observation
            # QUESTION - what for trials of different lengths?
            #   -> maybe deal with this in the fit/predict steps?
            t = poly.fit_transform((x[:,0]).reshape(-1,1))

            # only regress on data past reference time
            t = t[self.reference_time:]

            for column_index in np.arange(x.shape[1]):
                spot_index = column_index - 1
                if column_index == 0:
                    pass
                else:
                    # only fit data past the reference_time
                    # other data is 0 from preprocessing
                    model.fit(t, x[self.reference_time:,column_index])
                    coefficients[:, spot_index] = model.coef_
                    scores[spot_index] = model.score(t, x[self.reference_time:, column_index])
            coef_.iloc[trial_index] = coefficients
            scores_.iloc[trial_index] = scores

        end = time.time()
        ptf( 'Regressed %d trials, n=%d in %d seconds' % (len(X), self.n, (end-start)), self.logfile)
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
