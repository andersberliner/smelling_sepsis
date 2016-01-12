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
import time
from functools import partial

from utils_capstone import print_to_file_and_terminal as ptf

class SeriesModel(object):
    def __init__(self, logfile=None,
                    X=None, y=None,
                    color_scale = 'RGB',
                    color_vector_type = 'DI',
                    reference_time = 0,
                    max_time = 60, # 20 hrs max
                    min_time = 1, # time after ref_time to start fitting
                    use_last_timestep_results = False, # True for simple, False do to Bayesian approach
                    features_pickle = 'features.pkl',
                    featurizer_pickle = 'featurizer.pkl',
                    featurizer_coldstart = True,
                    reducer_pickle = 'reducer.pkl',
                    reducer_coldstart = True,
                    scaler_pickle = 'scaler.pkl',
                    scaler_coldstart = True,
                    detection_model='LR',
                    detection_model_arguments={},
                    detection_reducer='pca',
                    detection_reducer_arguments={},
                    detection_featurizer='poly',
                    detection_featurizer_arguments={},
                    gram_model='LR',
                    gram_model_arguments={},
                    gram_reducer='PCA',
                    gram_reducer_arguments={},
                    gram_featurizer='detection',
                    gram_featurizer_arguments={},
                    classification_model='LR',
                    classification_model_arguments={},
                    classification_reducer='PCA',
                    classification_reducer_arguments={},
                    classification_featurizer='gram',
                    classification_featurizer_arguments={}):
        self.X = X
        self.y = y

        self.logfile = logfile

        self.color_scale = color_scale
        self.color_vector_type = color_vector_type
        self.reference_time = reference_time
        self.max_time = max_time
        self.min_time = min_time

        self.use_last_timestep_results = use_last_timestep_results
        self.featurizer_pickle = featurizer_pickle
        self.featurizer_coldstart = featurizer_coldstart
        self.reducer_pickle = reducer_pickle
        self.reducer_coldstart = reducer_coldstart

        self.verbose = False
        self.trial_lengths = None
        self.number_of_columns = 220 # expected number of spots and colors

        # self.models is a group of models for detection, gram, classification
        # at each timepoint
        self.features = defaultdict(dict) # 10: [[np X n]]
        self.models = defaultdict(dict)
        self.featurizers = defaultdict(dict)
        self.scalers = default(dict)
        self.reducers = defaultdict(dict)
        self.times = []

        # set base models and reducers
        self.detection_base_model = detection_model
        self.detection_base_reducer = detection_reducer
        self.detection_base_featurizer = detection_featurizer

        self.gram_base_model = gram_model
        self.gram_base_reducer = gram_reducer
        self.gram_base_featurizer = gram_featurizer

        self.classification_base_model = classification_model
        self.classification_base_reducer = classification_reducer
        self.classification_base_featurizer = classification_featurizer

        self.detection_base_model_arguments = detection_model_arguments
        self.detection_base_reducer_arguments = detection_reducer_arguments
        self.detection_base_featurizer_arguments = detection_featurizer_arguments

        self.gram_base_model_arguments = gram_model_arguments
        self.gram_base_reducer_arguments = gram_reducer_arguments
        self.gram_base_featurizer_arguments = gram_featurizer_arguments

        self.classification_base_model_arguments = classification_model_arguments
        self.classification_base_reducer_arguments = classification_reducer_arguments
        self.classification_base_featurizer_arguments = classification_featurizer_arguments

    # def __repr__(self):
    #     print self

    def _build_confusion_labels(self, y):
        # set confusion labels and their order
        confusion_labels = {}
        for col in y.columns:
            groups = y[col].unique()
            groups.sort()
            confusion_labels[col] = groups

        # change order of classification confusion matrix to put control first
        a = list(confusion_labels['classification'])
        b = [a.pop(a.index('Control'))]
        b.extend(a)
        confusion_labels['classification'] = np.array(b)

        return confusion_labels

    def _build_results_dataframes(self, number_of_trials):
        # self.predictions = y.copy()
        # self.probabilities = y.copy()
        #
        # # this stores by trial for ease of looking at single trials
        # for col in y.columns:
        #     self.predictions[col] = self.predictions[col].apply(lambda x: [])
        #     self.probabilities[col] = self.probabilities[col].apply(lambda x: [])



        # self.results: stores by time-step the prediction and probas at each time_step
        df_columns = ['time']
        for result_type in ['predictions', 'probabilities']:
            for k in self.confusion_labels.keys():
                df_columns.append(k + '_' + result_type)

        self.results = pd.DataFrame(columns=df_columns)

        # up-to-date predictions and probabilities, by result_type
        self.predictions = {}
        self.probabilities = {}
        for k, v in self.confusion_labels.iteritems():
            # predictions['detection'] is a n_samples X 1 array of most recent
            #   predictions
            self.predictions[k] = np.zeros(number_of_trials)
            # probabilities['classification'] is a n_samples X n_classes of
            #   most recent probabilities of classification for each class
            self.probabilities[k] = np.zeros((number_of_trials, len(v)))



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

    def _prepare_data(self, X):
        Z = self.preprocess(X)

        self._build_results_dataframes(len(Z))
        self.trial_lengths = self.find_trial_lengths(Z)
        self.inspect_trial_shapes(Z)

        return Z

    def fit(self, X, y, verbose=False, debug=False):
        self.verbose = verbose
        self.confusion_labels = self._build_confusion_labels(y)

        # start with the second time
        t = self.reference_time + self.min_time
        # don't use results for the first model made
        use_last_timestep_results = False
        if debug:
            self.times = [30,40,50]
        else:
            self.times = np.arange(t, self.max_time, 1)

        self._build_results_dataframes(len(X))
        self.trial_lengths = self.find_trial_lengths(X)
        self.inspect_trial_shapes(X)

        # Check trial integrity
        if self.max_time > self.trial_lengths.min():
            ptf( '***FIT ERROR***', self.logfile)
            ptf( 'Minimum trial_length, %s, is less than max_time, %s' % (self.trial_lengths.min(), self.max_time), self.logfile)
            return

        # generate all features or load all features
        if self.featurizers_coldstart:
            X_preprocessed = self.preprocess(X)
            X_featurized = self.featurize(X_preprocessed, self.featurizer_pickle)
            X_scaled = self.scale(X_featurized, self.scaler_pickle)
            X_reduced = self.reduce(X_scaled, self.reducer_pickle)
            self.pickle_features(X_reduced, self.features_pickle)
        else:
            X = self.load_features(self.features_pickle)



        for t in self.times:
            start = time.time()

            if self.verbose:
                ptf( '\n\nTIMESTEP %d...' % t, self.logfile)
            self._fit_one_timestep(X, y, t, use_last_timestep_results)
            use_last_timestep_results = self.use_last_timestep_results

            # Bayes update
            if not self.use_last_timestep_results:
                self.bayes_update(t)

            end = time.time()
            if True:
                ptf( '\n----TIMESTEP %d took %d seconds----\n\n' % (t, (end-start)), self.logfile)


    def featurize(self, X, featurizer_pickle):
        start = time.time()
        ptf('\n>> Featurizing data ...', self.logfile)
        for t in self.times():
            # featurize, storing featurizers at each timestep
            if self.verbose:
                ptf( 'Featurizing nt=%d ...' % number_of_times, self.logfile)
            X_train = self._subset_data(X, number_of_times)
            X_detection, X_gram, X_classification = self._featurize_one_timestep(X_train, t)

            # convert to numpy arrays
            np_X_detection = self._pandas_to_numpy(X_detection)
            np_X_gram = self._pandas_to_numpy(X_gram)
            np_X_classification = self._pandas_to_numpy(X_classification)
            if self.debug:
                print 'Checking featurized shapes', np_X_detection.shape, np_X_gram.shape, np_X_classification.shape

            # store features
            self.features['detection'][t] = np_X_detection
            self.features['gram'][t] = np_X_gram
            self.features['classification'][t] = np_X_classification

        end = time.time()
        ptf('\n>> Featurizing completed (%s seconds) <<' % (end-start), self.logfile)

        # pickle featurizers
        start = time.time()
        featurizer_file_name = featurizer_pickle
        featurizer_file = open(featurizer_file_name, 'wb')
        ptf('\n>> Pickling featurizers to %s ...' % featurizer_file_name, self.logfile)
        pickle.dump(self.featurizers, featurizer_file, -1)
        featurizer_file.close()

        end = time.time()
        ptf('\n>> Pickling completed (%s seconds) <<' % (end-start), self.logfile)

        return self.features

    def scale(self, X, scaler_pickle):
        start = time.time()
        ptf('\n>> Scaling data...', self.logfile)
        for t in self.times():
            # featurize, storing featurizers at each timestep
            if self.verbose:
                ptf( 'Scaling nt=%d ...' % number_of_times, self.logfile)
            np_X_detection, np_X_gram, np_X_classification = self._scale_one_timestep(X, t)

            if self.debug:
                print 'Checking scaled shapes', np_X_detection.shape, np_X_gram.shape, np_X_classification.shape

            # store scaled features
            self.features['detection'][t] = np_X_detection
            self.features['gram'][t] = np_X_gram
            self.features['classification'][t] = np_X_classification

        end = time.time()
        ptf('\n>> Scaling completed (%s seconds) <<' % (end-start), self.logfile)

        # pickle scalers
        start = time.time()
        scaler_file_name = scaler_pickle
        scaler_file = open(scaler_file_name, 'wb')
        ptf('\n>> Pickling scalers to %s ...' % scaler_file_name, self.logfile)
        pickle.dump(self.scalers, scaler_file, -1)
        scaler_file.close()

        end = time.time()
        ptf('\n>> Pickling completed (%s seconds) <<' % (end-start), self.logfile)

        return self.features

    def reduce(self, X, reducer_pickle):
        start = time.time()
        ptf('\n>> Featurizing data ...', self.logfile)
        for t in self.times():
            # featurize, storing featurizers at each timestep
            if self.verbose:
                ptf( 'Reducing nt=%d ...' % number_of_times, self.logfile)
            np_X_detection, np_X_gram, np_X_classification = self._reduce_one_timestep(X, t)

            if self.debug:
                print 'Checking reduced shapes', np_X_detection.shape, np_X_gram.shape, np_X_classification.shape

            # store reduced features
            self.features['detection'][t] = np_X_detection
            self.features['gram'][t] = np_X_gram
            self.features['classification'][t] = np_X_classification

        end = time.time()
        ptf('\n>> Reducing completed (%s seconds) <<' % (end-start), self.logfile)

        # pickle reducers
        start = time.time()
        reducer_file_name = reducer_pickle
        reducer_file = open(reducer_file_name, 'wb')
        ptf('\n>> Pickling reducers to %s' % reducer_file_name, self.logfile)
        pickle.dump(self.reducers, reducer_file, -1)
        reducer_file.close()

        end = time.time()
        ptf('\n>> Pickling completed (%s seconds) <<' % (end-start), self.logfile)

        return self.features



    def pickle_features(self, features_pickle):
        # pickle features
        features_file_name = features_pickle
        features_file = open(features_file_name, 'wb')
        ptf('\n>> Pickling featurizers to %s' % features_file_name, self.logfile)
        pickle.dump(sm, features_file, -1)
        features_file.close()





    def bayes_update(self,t):
        # use Bayesian prior/posterior ideas to update predictions, probabilities
        # based on most recent timestep
        pass

    def preprocess(self, X):
        start = time.time()
        ptf('\n>> Preprocessing data...', self.logfile)

        X = X.copy()
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

        end = time.time()
        ptf('\n>> Prepocessing completed (%s seconds) <<' % (end-start), self.logfile)
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
            featurizer_arguments['reference_time'] = self.reference_time
            featurizer_arguments['logfile'] = self.logfile
            featurizer_arguments['verbose'] = self.verbose
            featurizer = PolynomialFeaturizer(**featurizer_arguments)
        elif featurizer_type == 'kink':
            featurizer_arguments['reference_time'] = self.reference_time
            featurizer_arguments['logfile'] = self.logfile
            featurizer_arguments['verbose'] = self.verbose
            pass
        elif featurizer_type == 'sigmoid':
            featurizer_arguments['reference_time'] = self.reference_time
            featurizer_arguments['logfile'] = self.logfile
            featurizer_arguments['verbose'] = self.verbose
            pass
        elif featurizer_type == 'forecast':
            featurizer_arguments['reference_time'] = self.reference_time
            featurizer_arguments['logfile'] = self.logfile
            featurizer_arguments['verbose'] = self.verbose
            pass
        elif featurizer_type == 'longitudinal':
            featurizer_arguments['reference_time'] = self.reference_time
            featurizer_arguments['logfile'] = self.logfile
            featurizer_arguments['verbose'] = self.verbose
            pass

        X_features, scores = featurizer.fit_transform(X_train)
        # print X_features.head()

        # need to flatten the features
        X_flat = X_features.apply(lambda x: x.flatten())
        return X_features, featurizer

    def _predict_featurize_class(self, X, featurizer):
        X_features, scores = featurizer.fit_transform(X)
        # print X_features.head()

        # need to flatten the features
        X_flat = X_features.apply(lambda x: x.flatten())
        return X_features

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
    def _featurize_one_timestep(self, X_train, number_of_times):
        # detection
        start = time.time()
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

        end = time.time()
        if self.verbose:
            ptf('... %d seconds' % (end-time))
        return X_detection, X_gram, X_classification

    def _scale_class(self, X):
        ss = StandardScaler()
        X_scaled = ss.fit_transform(X)

        return X_scaled, ss


    def _scale_one_timestep(self, X_train, number_of_times):
        start = time.time()
        # detection
        X_detection_scaled, detection_scaler = self._scale_class(X_train['detection'][number_of_times])
        self.scalers['detection'][number_of_times] = detection_scaler

        # gram
        X_gram_scaled, gram_scaler = self._scale_class(X_train['gram'][number_of_times])
        self.scalers['gram'][number_of_times] = gram_scaler

        # classification
        X_classification_scaled, classification_scaler = self._scale_class(X_train['classification'][number_of_times])
        self.scalers['classification'][number_of_times] = classification_scaler

        end = time.time()
        if self.verbose:
            ptf('... %d seconds' % (end-time))
        return X_detection_scaled, X_gram_scaled, X_classification_scaled

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

    def _reduce_class(self, X_train, reducer_type, reducer_arguments):
        X_features = X_train.copy()
        if not reducer_type:
            return X_features, None

        if reducer_type == 'pca':
            reducer = PCA(**reducer_arguments)
        else:
            ptf('*** Unknown reducer_type %s.  No transformations done ***' % reducer_type, self.logfile)

        X_features, scores = reducer.fit_transform(X_train)

        return X_features, featurizer

    def _reduce_one_timestep(self, X_train, number_of_times):
        start = time.time()
        # detection
        X_detection, detection_reducer = self._reduce_class(X_train,
                                self.detection_base_reducer,
                                self.detection_base_reducer_arguments)
        self.reducers['detection'][number_of_times] = detection_reducer
        # for efficiency, if not using different methods, could have
        # all 3 be the same
        # gram
        if self.gram_base_reducer == 'detection':
            X_gram = X_detection.copy()
            gram_reducer = detection_reducer
        else:
            X_gram, gram_reducer = self._reduce_class(X_train,
                                    self.gram_base_reducer,
                                    self.gram_base_reducer_arguments)
        self.reducers['gram'][number_of_times] = gram_reducer

        # classification
        if self.classification_base_reducer == 'detection':
            X_classification = X_detection.copy()
            classification_reducer = detection_reducer
        elif self.classification_base_reducer == 'gram':
            X_classification = X_gram.copy()
            classification_reducer = gram_reducer
        else:
            X_classification, classification_reducer = self._reduce_class(X_train,
                                    self.classification_base_reducer,
                                    self.classification_base_reducer_arguments)

        self.reducers['classification'][number_of_times] = classification_reducer

        end = time.time()
        if self.verbose:
            ptf('... %d seconds' % (end-time))

        return X_detection, X_gram, X_classification

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

    def _fit_one_timestep(self, X, y, number_of_times, use_last_timestep_results=False):
        # subset the data
        X_train = self._subset_data(X, number_of_times)

        # featurize
        ptf( 'Featurizing nt=%d ...' % number_of_times, self.logfile)
        X_detection, X_gram, X_classification = self._featurize(X_train, number_of_times)
        np_X_detection = self._pandas_to_numpy(X_detection)
        if use_last_timestep_results:
            # append most recent probabilities of growth (col 1)
            np_X_detection = np.hstack((np_X_detection,
                                        self.probabilities['detection'][:,1].reshape(-1,1)))
        # fit detection
        ptf( 'Training detection nt=%d ...' % number_of_times, self.logfile)
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
        ptf( 'Training gram nt=%d ...' % number_of_times, self.logfile)
        np_X_gram = self._pandas_to_numpy(X_gram)
        # print np_X_gram.shape
        # print y_predict_detection.shape
        # print y_probabilities_detection.shape
        # print y_probabilities_detection[:,1].shape
        np_X_gram = np.hstack((np_X_gram, y_probabilities_detection[:,1].reshape(-1,1)))
        if use_last_timestep_results:
            # append probas of n, p (not control)
            np_X_gram = np.hstack((np_X_gram, self.probabilities['gram'][:,:2]))
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
        ptf( 'Training classification nt=%d ...' % number_of_times, self.logfile)
        np_X_classification = self._pandas_to_numpy(X_classification)
        # print np_X_classification.shape
        # print y_probabilities_detection[:,1].reshape(-1,1).shape
        # print y_probabilities_gram[:,:2].shape
        np_X_classification = np.hstack((np_X_classification,
                            y_probabilities_detection[:,1].reshape(-1,1),
                            y_probabilities_gram[:,:-1]))
        if use_last_timestep_results:
            np_X_classification = np.hstack((np_X_classification,
                                        self.probabilities['classification'][:,:-1]))
        # classification_model = self.fit_classification(np_X_classification, y['classification'])
        classification_model = self._fit_class(np_X_classification, y['classification'],
                                self.classification_base_model,
                                self.classification_base_model_arguments,
                                step=('classification t=%d' % number_of_times))

        # store model
        self.models['classification'][number_of_times] = classification_model

        y_predict_classification = classification_model.predict(np_X_classification)
        y_probabilities_classification = classification_model.predict_proba(np_X_classification)

        # score and write to scores
        self._score_one_timestep(y, y_predict_detection,
                                 y_predict_gram, y_predict_classification,
                                 number_of_times)
        # store results from this timestep
        self._store_one_timestep((y_predict_detection, y_probabilities_detection),
                                 (y_predict_gram, y_probabilities_gram),
                                 (y_predict_classification, y_probabilities_classification),
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

    def predict_proba(self, X, verbose=False):
        yd, yg, yc = self.predict(X, verbose)

        return self.probabilities['detection'], self.probabilities['gram'], self.probabilities['classification']


    def predict(self, X, verbose=False):
        self.verbose = verbose

        # preprocesses X via manner used to fit
        # also creates scores, results, predictions and probabilities df
        # NOTE: this overwrites the existing df created during the fit step
        X = self._prepare_data(X)

        if self.max_time > self.trial_lengths.min():
            print '***FIT ERROR***'
            print 'Minimum trial_length, %s, is less than max_time, %s' % (self.trial_lengths.min(), self.max_time)
            return

        # step thru timesteps, featurizers, preprocessors, models and predicts
        use_last_timestep_results = False
        for t in self.times:
            start = time.time()

            self._predict_one_timestep(X,t,use_last_timestep_results)
            use_last_timestep_results = self.use_last_timestep_results

            # Bayes update
            if not self.use_last_timestep_results:
                self.bayes_update(t)

            end = time.time()
            if self.verbose:
                ptf( '\n----PREDICT TIMESTEP %d took %d seconds----\n\n' % (t, (end-start)), self.logfile)

        return self.predictions['detection'], self.predictions['gram'], self.predictions['classification']


    def _predict_one_timestep(self, X, number_of_times, use_last_timestep_results=False):
        # subset the data
        X = self._subset_data(X, number_of_times)

        # featurize
        X_detection, X_gram, X_classification = self._predict_featurize(X, number_of_times)

        # predict detection
        ptf( 'Predicting detection nt=%d ...' % number_of_times, self.logfile)
        np_X_detection = self._pandas_to_numpy(X_detection)

        # use last timestep if required
        if use_last_timestep_results:
            # append most recent probabilities of growth (col 1)
            np_X_detection = np.hstack((np_X_detection,
                                        self.probabilities['detection'][:,1].reshape(-1,1)))

        # predict detection
        detection_model = self.models['detection'][number_of_times]
        y_predict_detection = detection_model.predict(np_X_detection)
        y_probabilities_detection = detection_model.predict_proba(np_X_detection)

        # predict gram
        ptf( 'Predicting gram nt=%d ...' % number_of_times, self.logfile)
        np_X_gram = self._pandas_to_numpy(X_gram)
        np_X_gram = np.hstack((np_X_gram, y_probabilities_detection[:,1].reshape(-1,1)))

        if use_last_timestep_results:
            # append probas of n, p (not control)
            np_X_gram = np.hstack((np_X_gram, self.probabilities['gram'][:,:2]))

        gram_model = self.models['gram'][number_of_times]
        y_predict_gram = gram_model.predict(np_X_gram)
        y_probabilities_gram = gram_model.predict_proba(np_X_gram)

        # fit classification
        ptf( 'Training classification nt=%d ...' % number_of_times, self.logfile)
        np_X_classification = self._pandas_to_numpy(X_classification)
        np_X_classification = np.hstack((np_X_classification,
                            y_probabilities_detection[:,1].reshape(-1,1),
                            y_probabilities_gram[:,:-1]))
        if use_last_timestep_results:
            np_X_classification = np.hstack((np_X_classification,
                                        self.probabilities['classification'][:,:-1]))

        classification_model = self.models['classification'][number_of_times]

        y_predict_classification = classification_model.predict(np_X_classification)
        y_probabilities_classification = classification_model.predict_proba(np_X_classification)

        # store results from this timestep
        self._store_one_timestep((y_predict_detection, y_probabilities_detection),
                                 (y_predict_gram, y_probabilities_gram),
                                 (y_predict_classification, y_probabilities_classification),
                                 number_of_times)


    def _predict_featurize(self, X, number_of_times):
        detection_featurizer = self.featurizers['detection'][number_of_times]
        X_detection = self._predict_featurize_class(X, detection_featurizer)

        if self.gram_base_featurizer == 'detection':
            X_gram = X_detection.copy()
        else:
            gram_featurizer = self.featurizers['gram'][number_of_times]
            X_gram = self._predict_featurize_class(X, gram_featurizer)

        if self.classification_base_featurizer == 'detection':
            X_classification = X_detection.copy()
        elif self.classification_base_featurizer == 'gram':
            X_classification = X_gram.copy()
        else:
            classification_featurizer = self.featurizers['classification'][number_of_times]
            X_classification = self._predict_featurize_class(X, classification_featurizer)

        return X_detection, X_gram, X_classification



    def _get_column_value_by_time(self, df, column, time):
        return df[df['time']==time][column].values[0]

    def score(self, y, verbose=False):
        # NOTE - scores at each time-step
        # must follow a fit or predict step
        self.verbose = verbose

        for t in self.times:
            y_predict_detection = self._get_column_value_by_time(self.results, 'detection_predictions', t)
            y_predict_gram = self._get_column_value_by_time(self.results, 'gram_predictions', t)
            y_predict_classification = self._get_column_value_by_time(self.results, 'classification_predictions', t)

            y_probabilities_detection = self._get_column_value_by_time(self.results, 'detection_probabilities', t)
            y_probabilities_gram = self._get_column_value_by_time(self.results, 'gram_probabilities', t)
            y_probabilities_classification = self._get_column_value_by_time(self.results, 'classification_probabilities', t)

            # score and write to scores
            self._score_one_timestep(y, y_predict_detection,
                                     y_predict_gram, y_predict_classification,
                                     t)

            # store results from this timestep
            self._store_one_timestep((y_predict_detection, y_probabilities_detection),
                                     (y_predict_gram, y_probabilities_gram),
                                     (y_predict_classification, y_probabilities_classification),
                                     t)
        return self.results

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

    def _store_one_timestep(self, detection_tuple, gram_tuple, classification_tuple, number_of_times):
        results_dict = {}
        results_dict['time'] = number_of_times

        # update results data_frame
        results_tuples = [detection_tuple, gram_tuple, classification_tuple]
        results_types = ['detection', 'gram', 'classification']
        for (predictions, probabilities), results_type in izip(results_tuples, results_types):
            results_dict[results_type + '_' + 'predictions'] = predictions
            results_dict[results_type + '_' + 'probabilities'] = probabilities

        self._append_row_to_df(self.results, results_dict)

        # update predictions and probabilities
        results_tuples = [detection_tuple, gram_tuple, classification_tuple]
        results_types = ['detection', 'gram', 'classification']
        for (predictions, probabilities), results_type in izip(results_tuples, results_types):
            self.predictions[results_type] = predictions
            self.probabilities[results_type] = probabilities

    def _score_one_timestep(self, y_train, y_predict_detection,
                             y_predict_gram, y_predict_classification,
                             number_of_times):
        # detection - calculate results
        if self.verbose:
            ptf( 'Detection results', self.logfile)
            ptf( mcm.classification_report_ovr(y_train['detection'],
                y_predict_detection,
                self.confusion_labels['detection']), self.logfile)
            ptf( 'Gram results', self.logfile)
            ptf( mcm.classification_report_ovr(y_train['gram'],
                y_predict_gram,
                self.confusion_labels['gram']), self.logfile)
            ptf( 'Classification results', self.logfile)
            ptf( mcm.classification_report_ovr(y_train['classification'],
                y_predict_classification,
                self.confusion_labels['classification'], s1=30), self.logfile)

        scores = mcm.scores_binary(y_train['detection'], y_predict_detection)
        # print scores
        # builds confusion matrix of TP, FP, etc. for the detection case
        cm = mcm.confusion_matrix_binary(y_train['detection'], y_predict_detection)
        # print cm
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
            micros, macros = mcm.micro_macro_scores(results)

            # add results for each label
            for i, label in enumerate(labels):
                label_cm = mcm.confusion_matrix_ovr(*results[i,:])
                score_dict = self._populate_score_dict(label_cm, scores[i,:], number_of_times)
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
