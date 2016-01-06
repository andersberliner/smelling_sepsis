# seriesmodel.py
# Anders Berliner
# 20160105
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import izip

class SeriesModel(object):
    def __init__(self, model=LR, preprocessor=PCA):
        self.X = None
        self.y = None
        self.trial_lengths = None
        self.predictions = None
        self.scores = None
        self.model = model
        self.preprocessor = preprocessor
        self.verbose = False
        self.number_of_columns = 220 # expected number of spots and colors

    def __repr__(self):
        pass

    def fit(self, X, y, verbose=False):
        self.X = X
        self.y = y
        self.verbose = verbose

        self.trial_lengths = self.find_trial_lengths(self.X)
        self.inspect_trial_shapes(self.X)

        # start with the second time
        t = 1
        while t < self.trial_lengths.max():
            self._fit_one_timestep(t)
            t += 1


    def _fit_one_timestep(self, number_of_times):
        # subset the data
        X_train = self._subset_data(self.X, number_of_times)
        y_train = self._subset_data(self.y, number_of_times)

        # fit detection
        X_detection = self._featurize_detection(X_train)
        self.model.fit(X_detection, y_train['detection'])
        # predict detection and probabilities.
        #Use as features to fit gram
        y_predict_detection = self.model.predict(X_detection)
        y_probabilities_detection = self.model.predict_proba(X_detection)

        # fit gram
        X_gram = self._featurize_gram(X_train, y_predict_detection, y_probabilities_detection)
        self.model.fit(X_gram, y_train['gram'])
        # predict gram and probabilities.
        # Use as features to fit classification
        y_predict_gram = self.model.predict(X_gram)
        y_probabilities_gram = self.model.predict_proba(X_gram)

        # fit classification
        X_classification = self._featurize_classification(X_train, y_predict_gram, y_probabilities_gram)
        self.model.fit(X_classification, y_train['classification'])
        y_predict_classification = self.model.predict(X_classification)
        y_probabilities_classification = self.model.predict_proba(X_classification)

        # score
        y_predict = self.agglomerate_predictions(y_predict_detection, y_predict_gram, y_predict_classification)

        self.add_scores(y_predict, number_of_times)

    def _subset_data(self, X, number_of_times):
        pass

    def _featurize_detection(self, X_train):
        pass

    def _featurize_gram(self, X_train):
        pass

    def _featurize_classification(self, X_train):
        pass

    def predict(self, X):
        pass

    def _predict_one_timestep(self, X, number_of_times):
        pass

    def score(self):
        pass

    def _score_one_timestep(self):
        pass

    def find_trial_lengths(self, X):
        trial_lengths = np.zeros(len(X))
        for i, trial in enumerate(X):
            trial_lengths[i] = len(trial['data'])
        return trial_lengths

    def inspect_trial_shapes(self, X):
        trial_widths = np.zeros(len(X))
        for i, trial in enumerate(X):
            trial_widths[i] = shape(trial['data'])[1]

        trial_indexes = X.index.values
        trials_to_inspect = trial_indexes[trial_widths != self.number_of_columns]
        trial_widths_to_inspect = trial_widths[trial_widhts != self.number_of_columns]
        for index, width in izip(trials_to_inspect, trial_widths_to_inspect):
            print '**ERROR: Check trial %d - has %d columns' % (index, width)

if __name__ == '__main__':
    pass
