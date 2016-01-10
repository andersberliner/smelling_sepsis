# seriesmodel.py
# Anders Berliner
# 20160105
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import izip
from collections import defaultdict
from featurizer import PolynomialFeaturizer
import multiclassmetrics as mcm

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
                    gram_featurizer='detection',
                    gram_featurizer_arguments={},
                    classification_model='LR',
                    classification_model_arguments={},
                    classification_preprocessor='PCA',
                    classification_preprocessor_arguments={},
                    classification_featurizer='gram',
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
        self.models = defaultdict(dict)
        self.featurizers = defaultdict(dict)
        self.preprocessors = defaultdict(dict)

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

        self.detection_base_model_arguments = detection_model_arguments
        self.detection_base_preprocessor_arguments = detection_preprocessor_arguments
        self.detection_base_featurizer_arguments = detection_featurizer_arguments

        self.gram_base_model_arguments = gram_model_arguments
        self.gram_base_preprocessor_arguments = gram_preprocessor_arguments
        self.gram_base_featurizer_arguments = gram_featurizer_arguments

        self.classification_base_model_arguments = classification_model_arguments
        self.classification_base_preprocessor_arguments = classification_preprocessor_arguments
        self.classification_base_featurizer_arguments = classification_featurizer_arguments

    def __repr__(self):
        pass

    def _build_results_dataframes(self, y):
        self.predictions = y.copy()
        self.probabilities = y.copy()

        for col in y.columns:
            self.predictions[col] = self.predictions[col].apply(lambda x: [])
            self.probabilities[col] = self.probabilities[col].apply(lambda x: [])

        self.confusion_labels = {}
        for col in y.columns:
            groups = y[col].unique()
            groups.sort()
            self.confusion_labels[col] = groups

        # change order of classification confusion matrix to put control first
        a = list(self.confusion_labels['classification'])
        b = [a.pop(a.index('Control'))]
        b.extend(a)
        self.confusion_labels['classification'] = np.array(b)

        # for all of our metrics, build a pandas dataframe column
        self.scores = {}
        df_columns = ['time']
        self.metrics = ['confusion_matrix', 'accuracy', 'precision', 'recall', 'f1']
        df_columns.extend(self.metrics)
        for label, label_list in self.confusion_labels.iteritems():
            if label == 'detection':
                self.scores[label] = pd.DataFrame(columns=df_columns)
            else:
                self.scores[label] = {}
                # print label_list
                for k in label_list:
                    self.scores[label][k] = pd.DataFrame(columns=df_columns)
                self.scores[label]['micro'] = pd.DataFrame(columns=df_columns)
                self.scores[label]['macro'] = pd.DataFrame(columns=df_columns)

    def _prepare_data(self, X, y):
        Z = self.preprocess(X.copy())

        self._build_results_dataframes(y)
        self.trial_lengths = self.find_trial_lengths(Z)
        self.inspect_trial_shapes(Z)

        return Z

    def fit(self, X, y, verbose=False):
        self.verbose = verbose
        X = self._prepare_data(X,y)

        # start with the second time
        t = 1
        while t < self.trial_lengths.max():
            self._fit_one_timestep(X, y, t)
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


    def _featurize_class(self, X_train, featurizer_type, featurizer_arguments):
        X_features = X_train.copy()
        if not featurizer_type:
            return X_features

        if featurizer_type == 'poly':
            featurizer = PolynomialFeaturizer(**featurizer_arguments)
        elif featurizer_type == 'kink':
            pass
        elif featurizer_type == 'sigmoid':
            pass
        elif featurizer_type == 'forecast':
            pass
        elif featurizer_type == 'longitudinal':
            pass

        X_features, scores = featurizer.fit_transform(X_train)
        # print X_features.head()

        # need to flatten the features
        X_flat = X_features.apply(lambda x: x.flatten())
        return X_features, featurizer

    # def _featurize_gram(self, X_train):
    #     # featurizer = PolynomialFeaturizer()
    #     if self.detection_featurizer == 'poly':
    #         featurizer = PolynomialFeaturizer()
    #     X_gram = featurizer(X_train)
    #     return X_gram
    #
    # def _featurize_classification(self, X_train):
    #     # featurizer = PolynomialFeaturizer()
    #     X_classification = self.classification_featurizer(X_train)
    #     return X_classification

    def _featurize(self, X_train, number_of_times):
        # detection
        X_detection, detection_featurizer = self._featurize_class(X_train,
                                self.detection_base_featurizer,
                                self.detection_base_featurizer_arguments)
        self.featurizers['detection'][number_of_times] = detection_featurizer
        # for efficiency, if not using different methods, could have
        # all 3 be the same
        # gram
        if self.gram_base_featurizer == 'detection':
            X_gram = X_detection.copy()
            gram_featurizer = detection_featurizer
        else:
            X_gram, gram_featurizer = self._featurize_class(X_train,
                                    self.gram_base_featurizer,
                                    self.gram_base_featurizer_arguments)
        self.featurizers['gram'][number_of_times] = gram_featurizer

        # classification
        if self.classification_base_featurizer == 'detection':
            X_classification = X_detection.copy()
            classification_featurizer = detection_featurizer
        elif self.classification_base_featurizer == 'gram':
            X_classification = X_gram.copy()
            classification_featurizer = gram_featurizer
        else:
            X_classification, classification_featurizer = self._featurize_class(X_train,
                                    self.classification_base_featurizer,
                                    self.classification_base_featurizer_arguments)

        self.featurizers['classification'][number_of_times] = classification_featurizer

        return X_detection, X_gram, X_classification

    def _fit_class(self, X_train, y_train, model_type, model_argmuents, step=None):
        if model_type == 'LR':
            model = LogisticRegression(**model_argmuents)
        elif model_type == 'RF':
            pass
        elif model_type == 'SVM':
            pass
        elif model_type == 'longitudinal':
            pass
        else:
            print 'Invalid model_type %s for %s' % (model_type, step)

        model.fit(X_train, y_train)

        return model

    def _pandas_to_numpy(self, df_X):
        # features are in a pandas dataframe of the format
        #  trial    features
        #  1        [[f11, f12, ...,f1n],
        #            [f21,...],
        #            ....
        #            [fp1, fp2, ..., fpn]]
        #  2         number_spot_features X number_of_spots
        #
        # All sklearn need np array of shape
        #    number_trials X number_features
        np_X = [x.flatten() for x in df_X.values]
        np_X = tuple(np_X)
        np_X = np.vstack(np_X)

        return np_X

    def _fit_one_timestep(self, X, y, number_of_times):
        # subset the data
        X_train = self._subset_data(X, number_of_times)

        # featurize
        X_detection, X_gram, X_classification = self._featurize(X_train, number_of_times)
        np_X_detection = self._pandas_to_numpy(X_detection)
        # fit detection
        print 'Training detection nt=%d ...' % number_of_times
        # detection_model = self.fit_detection(X_detection, y['detection'])
        # print type(X), type(X_train), type(X_detection), type(np_X_detection)
        # print X.shape, X_train.shape, X_detection.shape, np_X_detection.shape
        # #print X_detection.shape, X_gram.shape, X_classification.shape
        # print type(y)
        # print y.shape
        detection_model = self._fit_class(np_X_detection,
                                y['detection'].values,
                                self.detection_base_model,
                                self.detection_base_model_arguments,
                                step=('detection t=%d' % number_of_times))

        # store model
        self.models['detection'][number_of_times] = detection_model

        # predict detection and probabilities.
        # Use as features to fit gram
        y_predict_detection = detection_model.predict(np_X_detection)
        y_probabilities_detection = detection_model.predict_proba(np_X_detection)

        # fit gram
        print 'Training gram nt=%d ...' % number_of_times
        np_X_gram = self._pandas_to_numpy(X_gram)
        # print np_X_gram.shape
        # print y_predict_detection.shape
        # print y_probabilities_detection.shape
        # print y_probabilities_detection[:,1].shape
        np_X_gram = np.hstack((np_X_gram, y_probabilities_detection[:,1].reshape(-1,1)))
        gram_model = self._fit_class(np_X_gram,
                                y['gram'].values,
                                self.gram_base_model,
                                self.gram_base_model_arguments,
                                step=('gram t=%d' % number_of_times))

        # store model
        self.models['gram'][number_of_times] = gram_model

        # predict gram and probabilities.
        # Use as features to fit classification
        y_predict_gram = gram_model.predict(np_X_gram)
        y_probabilities_gram = gram_model.predict_proba(np_X_gram)

        # fit classification
        print 'Training classification nt=%d ...' % number_of_times
        np_X_classification = self._pandas_to_numpy(X_classification)
        # print np_X_classification.shape
        # print y_probabilities_detection[:,1].reshape(-1,1).shape
        # print y_probabilities_gram[:,:2].shape
        np_X_classification = np.hstack((np_X_classification,
                            y_probabilities_detection[:,1].reshape(-1,1),
                            y_probabilities_gram[:,:2]))
        # classification_model = self.fit_classification(np_X_classification, y['classification'])
        classification_model = self._fit_class(np_X_classification, y['classification'],
                                self.classification_base_model,
                                self.classification_base_model_arguments,
                                step=('classification t=%d' % number_of_times))

        # store model
        self.models['classification'][number_of_times] = classification_model

        y_predict_classification = classification_model.predict(np_X_classification)
        y_probabilities_classification = classification_model.predict_proba(np_X_classification)

        # score and write to predictions, probabilities, scores
        self._score_one_timestep(y, y_predict_detection,
                                 y_predict_gram, y_predict_classification,
                                 number_of_times)

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

    def _populate_score_dict(self, cm, results, number_of_times):
        score_dict = {}
        score_dict['time'] = number_of_times
        score_dict['confusion_matrix'] = cm
        score_dict['accuracy'] = results[0]
        score_dict['precision'] = results[1]
        score_dict['recall'] = results[2]
        score_dict['f1'] = results[3]

        return score_dict

    def _append_row_to_df(self, df, row):
        df.loc[len(df)+1] = row

    def _score_one_timestep(self, y_train, y_predict_detection,
                             y_predict_gram, y_predict_classification,
                             number_of_times):
        # detection - calculate results
        if self.verbose:
            print 'Detection results'
            print mcm.classification_report_ovr(y_train['detection'], y_predict_detection, [0,1])


        scores = mcm.scores_binary(y_train['detection'], y_predict_detection)
        print scores
        # builds confusion matrix of TP, FP, etc. for the detection case
        cm = mcm.confusion_matrix_binary(y_train['detection'], y_predict_detection)
        print cm
        # detection - populate scores
        score_dict = self._populate_score_dict(cm, scores, number_of_times)
        # self.scores['detection'] = self.scores['detection'].append(score_dict, ignore_index=True)
        self._append_row_to_df(self.scores['detection'], score_dict)

        # gram, classification - calculate and populate results
        for result_type, predictions in izip(['gram', 'classification'],[y_predict_gram, y_predict_classification]):
            labels = list(self.confusion_labels[result_type])
            cm = mcm.confusion_matrix_mc(y_train[result_type], predictions, labels)
            results = mcm.results_ovr(y_train[result_type], predictions, labels)
            scores = mcm.scores_ovr(y_train[result_type], predictions, labels)
            micros, macros = mcm.micro_macro_results(results)

            # add results for each label
            for i, label in enumerate(labels):
                label_cm = mcm.confusion_matrix_ovr(*results[i,:])
                score_dict = self._populate_score_dict(cm, results[i,:], number_of_times)
                # self.scores[result_type][label].append(score_dict, ignore_index=True)
                self._append_row_to_df(self.scores[result_type][label], score_dict)

            score_dict = self._populate_score_dict(cm, micros, number_of_times)
            # self.scores[result_type]['micro'].append(score_dict, ignore_index=True)
            self._append_row_to_df(self.scores[result_type]['micro'], score_dict)
            score_dict = self._populate_score_dict(cm, macros, number_of_times)
            # self.scores[result_type]['macro'].append(score_dict, ignore_index=True)
            self._append_row_to_df(self.scores[result_type]['macro'], score_dict)

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
