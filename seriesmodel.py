# seriesmodel.py
# Anders Berliner
# 20160105
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import izip
from collections import defaultdict

class SeriesModel(object):
    def __init__(self, X=None, y=None,
            color_scale = 'RGB', color_vector_type = 'DI',
            reference_time = 0,
            detection_model=LR, detection_preprocessor=PCA,
            gram_model=LR, gram_preprocessor=PCA,
            classification_model=LR, classification_preprocessor=PCA):
        self.X = X
        self.y = y
        self.predictions = None
        self.scores = None

        self.color_scale = color_scale
        self.color_vector_type = color_vector_type
        self.reference_time = reference_time

        self.verbose = False
        self.trial_lengths = None
        self.number_of_columns = 220 # expected number of spots and colors

        # self.models is a group of models for detection, gram, classification
        # at each timepoint
        self.models = defaultdict(list)

        # set base models and preprocessors
        self.detection_base_model = detection_model
        self.detection_base_preprocessor = detection_preprocessor
        self.gram_base_model = gram_model
        self.gram_base_preprocessor = gram_preprocessor
        self.classification_base_model = classification_model
        self.classification_base_preprocessor = classification_preprocessor


    def __repr__(self):
        pass

    def fit(self, X, y, verbose=False):
        self.X = self.preprocess(X.copy())
        self.y = y
        self.verbose = verbose

        self.trial_lengths = self.find_trial_lengths(self.X)
        self.inspect_trial_shapes(self.X)

        # start with the second time
        t = 1
        while t < self.trial_lengths.max():
            self._fit_one_timestep(t)
            t += 1

    def preprocess(self, X):
        reference_time = self.reference_time
        # change color-scale as required
        # assume it's RGB
        if self.color_scale == 'CSV':
            X = self._rgb_to_csv(X)

        if self.color_vector_type == 'I':
            pass
        elif self.color_vector_type == 'DI':
            X = X.apply(lambda x: self._calculate_differences(x, reference_time))
        elif self.color_vector_type == 'DII':
            X = X.apply(lambda x: self._calculate_normalized_differences(x, reference_time))

        return X

    def _calculate_normalized_differences(self, x, reference_time):
        z = np.copy(x.astype(float))
        # DI(0<=reference_time) = 0
        # DI(t>reference_time) = (I(t) - I(reference_time))(I(reference_time))*100
        z[reference_time:, 1:] = (z[reference_time:, 1:] \
                                - z[reference_time, 1:]) \
                                / z.astype(float)[reference_time, 1:] \
                                * 100.0
        z[:reference_time, 1:] = 0
        return z

    def _calculate_differences(self, x, reference_time):
        z = np.copy(x)
        # DI(0<=reference_time) = 0
        # DI(t>reference_time) = I(t) - I(reference_time)
        z[reference_time:, 1:] = z[reference_time:, 1:] - \
                                    z[reference_time, 1:]
        z[:reference_time, 1:] = 0
        return z

    def _rgb_to_csv(self, X):
        return X



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

    def _subset_data(self, Z, number_of_times):
        z_sub = Z.copy()
        z_sub['data'] = z_sub['data'].apply(lambda x: x.iloc[0:number_of_times].values)
        return z_sub.values

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
            trial_lengths[i] = len(trial)
        return trial_lengths

    def inspect_trial_shapes(self, X):
        trial_widths = np.zeros(len(X))
        for i, trial in enumerate(X):
            trial_widths[i] = trial.shape[1]

        trial_indexes = X.index.values
        trials_to_inspect = trial_indexes[trial_widths != self.number_of_columns]
        trial_widths_to_inspect = trial_widths[trial_widths != self.number_of_columns]
        for index, width in izip(trials_to_inspect, trial_widths_to_inspect):
            print '**ERROR: Check trial %d - has %d columns' % (index, width)

if __name__ == '__main__':
    print 'Series Model imported'
