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
from featurizer import PolynomialFeaturizer

class SeriesModel(object):
    def __init__(self, X=None, y=None,
                    color_scale = 'RGB',
                    color_vector_type = 'DI',
                    reference_time = 0,
                    detection_model='LR',
                    detection_model_arguments={},
                    detection_preprocessor='pca',
                    detection_preprocessor_arguments={},
                    detection_featurizer='poly',
                    detection_featurizer_arguments={},
                    gram_model='LR',
                    gram_model_arguments={},
                    gram_preprocessor='PCA',
                    gram_preprocessor_arguments={},
                    gram_featurizer='poly',
                    gram_featurizer_arguments={},
                    classification_model='LR',
                    classification_model_arguments={},
                    classification_preprocessor='PCA',
                    classification_preprocessor_arguments={},
                    classification_featurizer='poly',
                    classification_featurizer_arguments={}):
        self.X = X
        self.y = y

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
        self.detection_base_featurizer = detection_featurizer

        self.gram_base_model = gram_model
        self.gram_base_preprocessor = gram_preprocessor
        self.gram_base_featurizer = gram_featurizer

        self.classification_base_model = classification_model
        self.classification_base_preprocessor = classification_preprocessor
        self.classification_base_featurizer = classification_featurizer

    def __repr__(self):
        pass

    def _build_results_dataframes(self):
        self.predictions = self.y.copy()
        self.probabilities = self.y.copy()

        for col in self.y.columns:
            self.predictions[col] = self.predictions[col].apply(lambda x: [])
            self.probabilities[col] = self.probabilities[col].apply(lambda x: [])

        # for all of our metrics, build a pandas dataframe column
        df_columns = ['time']
        self.metrics = ['confusion', 'accuracy', 'sensitivity', 'recall']
        for label_type in self.y.columns:
            for metric in self.metrics:
                df_columns.append(label_type + '_' + metric)
        self.scores = pd.DataFrame(columns=df_columns)

        self.confusion_labels = {}
        for col in self.y.columns:
            self.confusion_labels[col] = self.y[col].unique()

    def _prepare_data(self, X, y):
        self.X = self.preprocess(X.copy())
        self.y = y

        self._build_results_dataframes()
        self.trial_lengths = self.find_trial_lengths(self.X)
        self.inspect_trial_shapes(self.X)

    def fit(self, X, y, verbose=False):
        self.verbose = verbose
        self._prepare_data(X,y)

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


    def _featurize_detection(self, X_train):
        if self.detection_featurizer == 'poly':
            featurizer = PolynomialFeaturizer()
        X_detection = self.detection_featurizer(X_train)
        return X_detection

    def _featurize_gram(self, X_train):
        # featurizer = PolynomialFeaturizer()
        X_gram = self.gram_featurizer(X_train)
        return X_gram

    def _featurize_classification(self, X_train):
        # featurizer = PolynomialFeaturizer()
        X_classification = self.classification_featurizer(X_train)
        return X_classification

    def _featurize(self, X_train):
        X_detection = self._featurize_detection(X_train)
        # for efficiency, if not using different methods, could have
        # all 3 be the same
        X_gram = self._featurize_gram(X_train)
        X_classification = self._featurize_classification(X_train)

        return X_detection, X_gram, X_classification

    def _fit_one_timestep(self, number_of_times):
        # subset the data
        X_train = self._subset_data(self.X, number_of_times)
        y_train = self._subset_data(self.y, number_of_times)

        # featurize
        X_detection, X_gram, X_classification = self._featurize(X_train)

        # fit detection
        detection_model = self.fit_detection(X_detection, y_train['detection'])

        # predict detection and probabilities.
        # Use as features to fit gram
        y_predict_detection = detection_model.predict(X_detection)
        y_probabilities_detection = detection_model.predict_proba(X_detection)

        # fit gram
        X_gram = np.hstack((X_gram, y_predict_detection.reshape(-1,1),
                            y_probabilities_detection[:,1].reshape(-1,1)))
        gram_model = self.fit_gram(X_gram, y_train['gram'])

        # predict gram and probabilities.
        # Use as features to fit classification
        y_predict_gram = gram_model.predict(X_gram)
        y_probabilities_gram = gram_model.predict_proba(X_gram)

        # fit classification
        X_classification = X_gram = np.hstack((X_classification,
                            y_predict_detection.reshape(-1,1),
                            y_probabilities_detection[:,1].reshape(-1,1),
                            y_predict_gram.reshape(-1,1),
                            y_probabilities_gram[:,2]))
        classification_model = self.fit_classification(X_classification, y_train['classification'])
        y_predict_classification = classification_model.predict(X_classification)
        y_probabilities_classification = classification_model.predict_proba(X_classification)

        # score and write to predictions, probabilities, scores
        self.score(number_of_times, y_predict_detection, y_predict_gram, y_predict_classification)

    def _subset_data(self, Z, number_of_times):
        z_sub = Z.copy()
        z_sub= z_sub.apply(lambda x: x[0:number_of_times])
        return z_sub

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

    def score(self, number_of_times, y_predict_detection, y_predict_gram, y_predict_classification):
        confusion = {}
        accuracy = {}

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
