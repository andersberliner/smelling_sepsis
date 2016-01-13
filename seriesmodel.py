# seriesmodel.py
# Anders Berliner
# 20160105
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from itertools import izip
from collections import defaultdict
from featurizer import PolynomialFeaturizer
import multiclassmetrics as mcm
import time
from functools import partial
import pickle

from output_capstone import print_to_file_and_terminal as ptf

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
                    fold_features_pickle = 'fold_features.pkl',
                    fold_features_test_pickle = 'fold_features_test.pkl',
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
                    gram_reducer='pca',
                    gram_reducer_arguments={},
                    gram_featurizer='detection',
                    gram_featurizer_arguments={},
                    classification_model='LR',
                    classification_model_arguments={},
                    classification_reducer='pca',
                    classification_reducer_arguments={},
                    classification_featurizer='gram',
                    classification_featurizer_arguments={},
                    nfolds=1,
                    fold_size=0.1):
        self.X = X
        self.y = y

        self.logfile = logfile

        # preprocessing parameters
        self.color_scale = color_scale
        self.color_vector_type = color_vector_type
        self.reference_time = reference_time
        self.max_time = max_time
        self.min_time = min_time

        # fit conditions
        self.use_last_timestep_results = use_last_timestep_results
        self.nfolds = nfolds
        self.fold_size = 0.1

        # data storage conditions
        self.featurizer_pickle = featurizer_pickle
        self.featurizer_coldstart = featurizer_coldstart
        self.reducer_pickle = reducer_pickle
        self.reducer_coldstart = reducer_coldstart
        self.scaler_pickle = scaler_pickle
        self.scaler_coldstart = scaler_coldstart
        self.features_pickle = features_pickle
        self.fold_features_pickle = fold_features_pickle
        self.fold_features_test_pickle = fold_features_test_pickle

        # other run conditions
        self.verbose = False
        self.trial_lengths = None
        self.number_of_columns = 220 # expected number of spots and colors + time

        # self.models is a group of models for detection, gram, classification
        # at each timepoint
        self.features = defaultdict(dict) # 10: [[np X n]]

        self.featurizers = defaultdict(dict)

        # these are commented since now they get filled in cross_val method
        # self.models = defaultdict(dict)
        # self.scalers = defaultdict(dict)
        # self.reducers = defaultdict(dict)
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

    ### MAIN METHODS ###

    # i) SETUP #
    def setup(self, X, y):
        start = time.time()
        ptf('\n>> i. Setting-up SeriesModel ...', self.logfile)
        self.confusion_labels = self._build_confusion_labels(y)
        self._build_results_dataframes(len(X))
        self.trial_lengths = self.find_trial_lengths(X)
        self.inspect_trial_shapes(X)
        self._build_crossvalidation_folds(y)

        end = time.time()
        ptf('\n>> Set-up completed (%s seconds) <<' % (end-start), self.logfile)

    # 0) PREPROCESS #
    def preprocess(self, X):
        start = time.time()
        ptf('\n>> 0. Preprocessing data ...', self.logfile)

        X = X.copy()
        reference_time = self.reference_time
        # change color-scale as required
        # assume it's RGB
        if self.color_scale == 'HSV':
            X = self._rgb_to_hsv(X)

        if self.color_vector_type == 'I':
            pass
        elif self.color_vector_type == 'DI':
            X = X.apply(lambda x: self._calculate_differences(x, reference_time))
        elif self.color_vector_type == 'DII':
            X = X.apply(lambda x: self._calculate_normalized_differences(x, reference_time))

        end = time.time()
        ptf('\n>> Prepocessing completed (%s seconds) <<' % (end-start), self.logfile)
        return X

    # 1) FEATURIZE #
    def featurize(self, X, featurizer_pickle):
        start = time.time()
        ptf('\n>> 1. Featurizing data ...', self.logfile)
        for t in self.times:
            number_of_times = t
            # featurize, storing featurizers at each timestep
            if self.verbose:
                ptf( 'Featurizing nt=%d ...' % number_of_times, self.logfile)

            X_train = self._subset_data(X, number_of_times)
            if self.debug:
                print t, X.iloc[0].shape
            X_detection, X_gram, X_classification = self._featurize_one_timestep(X_train, t)
            if self.debug:
                print t, X_detection.iloc[0].shape, X_gram.iloc[0].shape, X_classification.iloc[0].shape
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

    # 2) SCALE #
    def scale(self, X, scaler_pickle):
        start = time.time()
        ptf('\n>> 2. Scaling data ...', self.logfile)
        for fold, index_dict in self.folds.iteritems():
            # select only X_train
            # X_train = X
            self._scale_one_fold(X, fold)
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

        return self.fold_features

    # 3) REDUCE #
    def reduce(self, X, reducer_pickle):
        # pickle reducers
        start = time.time()
        ptf('\n>> 3. Reducing data ...', self.logfile)
        for fold in self.folds.keys():
            self._reduce_one_fold(X, fold)

        start = time.time()
        reducer_file_name = reducer_pickle
        reducer_file = open(reducer_file_name, 'wb')
        ptf('\n>> Pickling reducers to %s' % reducer_file_name, self.logfile)
        pickle.dump(self.reducers, reducer_file, -1)
        reducer_file.close()

        end = time.time()
        ptf('\n>> Pickling completed (%s seconds) <<' % (end-start), self.logfile)

        return self.fold_features

    # 4) PICKLE/LOAD FEATURES #
    def pickle_features(self, data, features_pickle):
        # pickle features
        features_file_name = features_pickle
        features_file = open(features_file_name, 'wb')
        pickle.dump(data, features_file, -1)
        features_file.close()

    def load_features(self, features_pickle):
        features_file_name = features_pickle
        features_file = open(features_file_name, 'rb')
        data = pickle.load(features_file)
        features_file.close()
        return data

    # 5) TRAIN #
    def train(self, X, y, fold, t, use_last_timestep_results=False):
        number_of_times = t
        (X_train_detection, X_train_gram, X_train_classification) = X
        (y_train_detection, y_train_gram, y_train_classification) = y

        # fit detection
        if self.verbose:
            ptf( 'Training detection fold:%d, nt:%d ...' % (fold, number_of_times), self.logfile)

        if use_last_timestep_results:
            # append most recent probabilities of growth (col 1)
            np_X_detection = np.hstack((X_train_detection,
                self.fold_probabilities[fold]['detection'][:,1].reshape(-1,1)))
        else:
            np_X_detection = X_train_detection

        model_detection = self._fit_class(np_X_detection,
            y_train_detection,
            self.detection_base_model,
            self.detection_base_model_arguments,
            step=('detection t=%d_%d' % (fold,number_of_times)))

        # store model, predict
        self.models[fold]['detection'][number_of_times] = model_detection
        y_predict_detection = model_detection.predict(np_X_detection)
        y_probabilities_detection = model_detection.predict_proba(np_X_detection)

        # fit gram
        if self.verbose:
            ptf( 'Training gram fold:%d, nt=%d ...' % (fold, number_of_times), self.logfile)

        np_X_gram = np.hstack((X_train_gram,
            y_probabilities_detection[:,1].reshape(-1,1)))
        if use_last_timestep_results:
            # append probas of n, p (not control)
            np_X_gram = np.hstack((np_X_gram,
                self.fold_probabilities[fold]['gram'][:,:2]))

        model_gram = self._fit_class(np_X_gram,
            y_train_gram,
            self.gram_base_model,
            self.gram_base_model_arguments,
            step=('gram t=%d_%d' % (fold, number_of_times)))

        # store model, predict
        self.models[fold]['gram'][number_of_times] = model_gram
        y_predict_gram = model_gram.predict(np_X_gram)
        y_probabilities_gram = model_gram.predict_proba(np_X_gram)

        # fit classification
        if self.verbose:
            ptf( 'Training classification fold:%d, nt=%d ...' % (fold, number_of_times), self.logfile)

        np_X_classification = np.hstack((X_train_classification,
            y_probabilities_detection[:,1].reshape(-1,1),
            y_probabilities_gram[:,:-1]))

        if use_last_timestep_results:
            # append probas of all non-control classes
            np_X_classification = np.hstack((np_X_classification,
                self.fold_probabilities[fold]['classification'][:,1:]))

        model_classification = self._fit_class(np_X_classification,
            y_train_classification,
            self.classification_base_model,
            self.classification_base_model_arguments,
            step=('classification t=%d_%d' % (fold, number_of_times)))

        # store model, predict
        self.models[fold]['classification'][number_of_times] = model_classification
        y_predict_classification = model_classification.predict(np_X_classification)
        y_probabilities_classification = model_classification.predict_proba(np_X_classification)

        models = (model_detection, model_gram, model_classification)
        predictions = (y_predict_detection, y_predict_gram, y_predict_classification)
        probabilities = (y_probabilities_detection, y_probabilities_gram, y_probabilities_classification)

        return models, predictions, probabilities

    # 6) PREDICT #
    def predict(self, models, X_test, fold, t, use_last_timestep_results):
        number_of_times = t
        (model_detection, model_gram, model_classification) = models
        (X_test_detection, X_test_gram, X_test_classification) = X_test

        if self.verbose:
            ptf( 'Predicting detection fold:%d, nt:%d ...' % (fold, number_of_times), self.logfile)

        if use_last_timestep_results:
            # append most recent probabilities of growth (col 1)
            np_X_detection = np.hstack((X_test_detection,
                self.fold_probabilities_test[fold]['detection'][:,1].reshape(-1,1)))
        else:
            np_X_detection = X_test_detection

        y_predict_detection = model_detection.predict(np_X_detection)
        y_probabilities_detection = model_detection.predict_proba(np_X_detection)

        # predict gram
        if self.verbose:
            ptf( 'Predciting gram fold:%d, nt=%d ...' % (fold, number_of_times), self.logfile)

        np_X_gram = np.hstack((X_test_gram,
            y_probabilities_detection[:,1].reshape(-1,1)))
        if use_last_timestep_results:
            # append probas of n, p (not control)
            np_X_gram = np.hstack((np_X_gram,
                self.fold_probabilities_test[fold]['gram'][:,:2]))

        y_predict_gram = model_gram.predict(np_X_gram)
        y_probabilities_gram = model_gram.predict_proba(np_X_gram)

        # predict classification
        if self.verbose:
            ptf( 'Predicting classification fold:%d, nt=%d ...' % (fold, number_of_times), self.logfile)

        np_X_classification = np.hstack((X_test_classification,
            y_probabilities_detection[:,1].reshape(-1,1),
            y_probabilities_gram[:,:-1]))

        if use_last_timestep_results:
            # append probas of all non-control classes
            np_X_classification = np.hstack((np_X_classification,
                self.probabilities[fold]['classification'][:,1:]))

        y_predict_classification = model_classification.predict(np_X_classification)
        y_probabilities_classification = model_classification.predict_proba(np_X_classification)

        predictions = (y_predict_detection, y_predict_gram, y_predict_classification)
        probabilities = (y_probabilities_detection, y_probabilities_gram, y_probabilities_classification)

        return predictions, probabilities

    # 7) STORE #

    # 8) SCORE #

    ### HELPER METHODS ###

    ## one_fold methods ##

    # 1) FEATURIZE #

    # 2) SCALE #
    def _scale_one_fold(self, X, fold):
        start = time.time()
        if self.verbose:
            ptf('\n> Scaling fold %d ...' % fold, self.logfile)

        for t in self.times:
            # featurize, storing featurizers at each timestep
            number_of_times = t
            if self.verbose:
                ptf( 'Scaling nt=%d ...' % number_of_times, self.logfile)
            np_X_detection, np_X_gram, np_X_classification = self._scale_one_timestep(X, t, fold)

            if self.debug:
                print 'Checking scaled shapes', np_X_detection[0].shape, np_X_gram[0].shape, np_X_classification[0].shape

            # store scaled features
            self.fold_features[fold]['detection'][t] = np_X_detection[0]
            self.fold_features[fold]['gram'][t] = np_X_gram[0]
            self.fold_features[fold]['classification'][t] = np_X_classification[0]

            self.fold_features_test[fold]['detection'][t] = np_X_detection[1]
            self.fold_features_test[fold]['gram'][t] = np_X_gram[1]
            self.fold_features_test[fold]['classification'][t] = np_X_classification[1]

        end = time.time()

        if self.verbose:
            ptf('\n> Scaling fold %d completed (%s seconds) <<' % (fold,(end-start)), self.logfile)

    # 3) REDUCE #
    def _reduce_one_fold(self, X, fold):
        start = time.time()
        if self.verbose:
            ptf('\n> Reducing fold %d ...' % fold, self.logfile)
        for t in self.times:
            number_of_times = t
            # featurize, storing featurizers at each timestep
            if self.verbose:
                ptf( 'Reducing nt=%d ...' % number_of_times, self.logfile)
            np_X_detection, np_X_gram, np_X_classification = self._reduce_one_timestep(X[fold], t, fold)

            if self.debug:
                print 'Checking reduced shapes', np_X_detection[0].shape, np_X_gram[0].shape, np_X_classification[0].shape

            # store reduced features
            self.fold_features[fold]['detection'][t] = np_X_detection[0]
            self.fold_features[fold]['gram'][t] = np_X_gram[0]
            self.fold_features[fold]['classification'][t] = np_X_classification[0]

            self.fold_features_test[fold]['detection'][t] = np_X_detection[1]
            self.fold_features_test[fold]['gram'][t] = np_X_gram[1]
            self.fold_features_test[fold]['classification'][t] = np_X_classification[1]

        end = time.time()
        if self.verbose:
            ptf('\n> Reducing fold %d completed (%s seconds) <<' % (fold,(end-start)), self.logfile)

    # 4) PICKLE/LOAD FEATURES #

    # 5) TRAIN #

    # 6) PREDICT #

    # 7) STORE #

    # 8) SCORE #

    ## one_timestep methods ##

    # 1) FEATURIZE #
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
            ptf('... %d seconds' % (end-start))
        return X_detection, X_gram, X_classification

    # 2) SCALE #
    def _subset_fold(self, X, fold, result_type, number_of_times, test_train='train'):
        X_test_train = X[result_type][number_of_times]
        X_test_train = X_test_train[self.folds[fold][test_train]]
        return X_test_train

    def _scale_one_timestep(self, X, number_of_times, fold):
        start = time.time()

        # pick just the data we need for this fold

        # X_train_detection = X_train['detection'][number_of_times]
        # X_train_detection = X_train_detection[self.folds[fold]['train']]

        # detection
        X_train_detection = self._subset_fold(X, fold, 'detection', number_of_times, 'train')
        X_detection_scaled, detection_scaler = self._scale_class(X_train_detection)
        self.scalers[fold]['detection'][number_of_times] = detection_scaler

        X_test_detection = self._subset_fold(X, fold, 'detection', number_of_times, 'test')
        X_test_detection_scaled = detection_scaler.transform(X_test_detection)

        # gram
        X_train_gram = self._subset_fold(X, fold, 'gram', number_of_times, 'train')
        X_gram_scaled, gram_scaler = self._scale_class(X_train_gram)
        self.scalers[fold]['gram'][number_of_times] = gram_scaler

        X_test_gram = self._subset_fold(X, fold, 'gram', number_of_times, 'test')
        X_test_gram_scaled = gram_scaler.transform(X_test_gram)

        # classification
        X_train_classification = self._subset_fold(X, fold, 'classification', number_of_times, 'train')
        X_classification_scaled, classification_scaler = self._scale_class(X_train_classification)
        self.scalers[fold]['classification'][number_of_times] = classification_scaler

        X_test_classification = self._subset_fold(X, fold, 'classification', number_of_times, 'test')
        X_test_classification_scaled = classification_scaler.transform(X_test_classification)

        end = time.time()
        if self.verbose:
            ptf('... %d seconds' % (end-start))

        X_scaleds = [(X_detection_scaled, X_test_detection_scaled),
            (X_gram_scaled, X_test_gram_scaled),
            (X_classification_scaled, X_test_classification_scaled)]
        return X_scaleds

    # 3) REDUCE #
    def _reduce_one_timestep(self, X_train, number_of_times, fold):
        start = time.time()
        # detection
        X_detection_reduced, detection_reducer = self._reduce_class(
                X_train['detection'][number_of_times],
                self.detection_base_reducer,
                self.detection_base_reducer_arguments)
        self.reducers[fold]['detection'][number_of_times] = detection_reducer

        X_detection_test = self.fold_features_test[fold]['detection'][number_of_times]
        X_detection_test_reduced = detection_reducer.transform(X_detection_test)
        # for efficiency, if not using different methods, could have
        # all 3 be the same
        # gram
        if self.gram_base_reducer == 'detection':
            X_gram_reduced = X_detection_reduced.copy()
            gram_reducer = detection_reducer
        else:
            X_gram_reduced, gram_reducer = self._reduce_class(
                    X_train['gram'][number_of_times],
                    self.gram_base_reducer,
                    self.gram_base_reducer_arguments)
        self.reducers[fold]['gram'][number_of_times] = gram_reducer

        X_gram_test = self.fold_features_test[fold]['gram'][number_of_times]
        X_gram_test_reduced = gram_reducer.transform(X_gram_test)

        # classification
        if self.classification_base_reducer == 'detection':
            X_classification_reduced = X_detection_reduced.copy()
            classification_reducer = detection_reducer
        elif self.classification_base_reducer == 'gram':
            X_classification_reduced = X_gram_reduced.copy()
            classification_reducer = gram_reducer
        else:
            X_classification_reduced, classification_reducer = self._reduce_class(
                    X_train['classification'][number_of_times],
                    self.classification_base_reducer,
                    self.classification_base_reducer_arguments)
        self.reducers[fold]['classification'][number_of_times] = classification_reducer

        X_classification_test = self.fold_features_test[fold]['classification'][number_of_times]
        X_classification_test_reduced = classification_reducer.transform(X_classification_test)

        end = time.time()
        if self.verbose:
            ptf('... %d seconds' % (end-start))

        X_reduceds = [(X_detection_reduced, X_detection_test_reduced),
            (X_gram_reduced, X_gram_test_reduced),
            (X_classification_reduced, X_classification_test_reduced)]
        return X_reduceds

    # 4) PICKLE/LOAD FEATURES #

    # 5) TRAIN #

    # 6) PREDICT #

    # 7) STORE #
    def _store_one_fold(self, train, test, fold, t):
        results_dict = {}
        results_dict['time'] = t
        results_dict['fold'] = fold
        labels_types = ['detection', 'gram', 'classification']
        testtrain_types = ['test', 'train']
        for testtrain, testtrain_type in izip((test, train), testtrain_types):
            (predictions, probabilities) = testtrain

            for class_predictions, class_probabilities, label_type in izip(predictions, probabilities, labels_types):
                results_dict[testtrain_type + '_' + label_type + '_' + 'predictions'] = class_predictions
                results_dict[testtrain_type + '_' + label_type + '_' + 'probabilities'] = class_probabilities

        self._append_row_to_df(self.results, results_dict)

        # update fold predicitions and probabilities
        for testtrain, testtrain_type in izip((test, train), testtrain_types):
            (predictions, probabilities) = testtrain

            for class_predictions, class_probabilities, label_type in izip(predictions, probabilities, labels_types):
                if testtrain_type == 'train':
                    self.fold_predictions[fold][label_type] = class_predictions
                    self.fold_probabilities[fold][label_type] = class_predictions
                else:
                    self.fold_predictions_test[fold][label_type] = class_predictions
                    self.fold_probabilities_test[fold][label_type] = class_predictions

    # 8) SCORE #
    #
    # def _score_one_timestep(self, y, y_predict_detection,
    #                          y_predict_gram, y_predict_classification,
    #                          number_of_times, fold):
    def _score_one_timestep(self, results_dicts, number_of_times):
        (y_train_true_timestep,
            y_train_predict_timestep,
            y_test_true_timestep,
            y_test_predict_timestep) = results_dicts

        # [y_train_detection, y_train_gram, y_train_classification] = y
        # detection - calculate results
        ytp_pairs = ((y_train_true_timestep, y_train_predict_timestep),
            (y_train_true_timestep, y_train_predict_timestep))
        for (yt,yp), testtrain_type in izip(ytp_pairs, ['train', 'test']):
            y_train_detection = yt['detection']
            y_train_gram = yt['gram']
            y_train_classification = yt['classification']
            y_predict_detection = yp['detection']
            y_predict_gram = yp['gram']
            y_predict_classification = yp['classification']

            if self.verbose:
                # print np.unique(y_predict_detection)
                # print np.unique(y_train_detection)
                # print self.confusion_labels['detection']
                ptf( '%s Detection results' % testtrain_type, self.logfile)
                ptf( mcm.classification_report_ovr(y_train_detection,
                    y_predict_detection,
                    self.confusion_labels['detection']), self.logfile)
                ptf( '%s Gram results' % testtrain_type, self.logfile)
                ptf( mcm.classification_report_ovr(y_train_gram,
                    y_predict_gram,
                    self.confusion_labels['gram']), self.logfile)
                ptf( '%s Classification results' % testtrain_type, self.logfile)
                ptf( mcm.classification_report_ovr(y_train_classification,
                    y_predict_classification,
                    self.confusion_labels['classification'], s1=30), self.logfile)

            scores = mcm.scores_binary(y_train_detection, y_predict_detection)
            # print scores
            # builds confusion matrix of TP, FP, etc. for the detection case
            cm = mcm.confusion_matrix_binary(y_train_detection, y_predict_detection)
            # print cm
            # detection - populate scores
            score_dict = self._populate_score_dict(cm, scores, number_of_times)
            # self.scores['detection'] = self.scores['detection'].append(score_dict, ignore_index=True)

            # figure out how to append one set of scores
            if testtrain_type == 'train':
                self._append_row_to_df(self.scores['detection'], score_dict)
            else:
                self._append_row_to_df(self.scores_test['detection'], score_dict)


            # gram, classification - calculate and populate results
            for result_type, predictions, truths in izip(['gram', 'classification'],
                    [y_predict_gram, y_predict_classification],
                    [y_train_gram, y_train_classification]):
                labels = list(self.confusion_labels[result_type])
                cm = mcm.confusion_matrix_mc(truths, predictions, labels)
                results = mcm.results_ovr(truths, predictions, labels)
                scores = mcm.scores_ovr(truths, predictions, labels)
                micros, macros = mcm.micro_macro_scores(results)

                # add results for each label
                for i, label in enumerate(labels):
                    label_cm = mcm.confusion_matrix_ovr(*results[i,:])
                    score_dict = self._populate_score_dict(label_cm, scores[i,:], number_of_times)
                    # self.scores[result_type][label].append(score_dict, ignore_index=True)
                    if testtrain_type == 'train':
                        self._append_row_to_df(self.scores[result_type][label], score_dict)
                    else:
                        self._append_row_to_df(self.scores_test[result_type][label], score_dict)

                score_dict = self._populate_score_dict(cm, micros, number_of_times)
                score_dict = self._populate_score_dict(cm, macros, number_of_times)

                if testtrain_type == 'train':
                    # self.scores[result_type]['micro'].append(score_dict, ignore_index=True)
                    self._append_row_to_df(self.scores[result_type]['micro'], score_dict)

                    # self.scores[result_type]['macro'].append(score_dict, ignore_index=True)
                    self._append_row_to_df(self.scores[result_type]['macro'], score_dict)
                else:
                    self._append_row_to_df(self.scores_test[result_type]['micro'], score_dict)
                    self._append_row_to_df(self.scores_test[result_type]['macro'], score_dict)


    ## one_class methods ##

    # 1) FEATURIZE #
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

    # 2) SCALE #
    def _scale_class(self, X):
        ss = StandardScaler()
        X = X.copy()
        X_scaled = ss.fit_transform(X)

        return X_scaled, ss

    # 3) REDUCE #
    def _reduce_class(self, X_train, reducer_type, reducer_arguments):
        X_features = X_train.copy()
        if not reducer_type:
            return X_features, None

        if reducer_type == 'pca':
            reducer = PCA(**reducer_arguments)
        else:
            ptf('*** Unknown reducer_type %s.  No transformations done ***' % reducer_type, self.logfile)

        X_features = reducer.fit_transform(X_train)

        return X_features, reducer

    # 4) PICKLE/LOAD FEATURES #

    # 5) TRAIN #

    # 6) PREDICT #
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

    # 7) STORE #

    # 8) SCORE #

    ### SET-UP METHODS ###

    # i. SETUP #
    def _build_results_dataframes(self, number_of_trials):
        # self.predictions = y.copy()
        # self.probabilities = y.copy()
        #
        # # this stores by trial for ease of looking at single trials
        # for col in y.columns:
        #     self.predictions[col] = self.predictions[col].apply(lambda x: [])
        #     self.probabilities[col] = self.probabilities[col].apply(lambda x: [])



        # self.results: stores by time-step the prediction and probas at each
        # time_step and fold for test and train
        df_columns = ['time', 'fold']
        for testtrain_type in ['test', 'train']:
            for label_type in self.confusion_labels.keys():
                for result_type in ['predictions', 'probabilities']:
                    cname = testtrain_type + '_' + label_type + '_' + result_type
                    df_columns.append(cname)

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
        self.scores_test = {}
        df_columns = ['time']
        self.metrics = ['confusion_matrix', 'accuracy', 'precision', 'recall', 'f1']
        df_columns.extend(self.metrics)
        for label, label_list in self.confusion_labels.iteritems():
            if label == 'detection':
                self.scores[label] = pd.DataFrame(columns=df_columns)
                self.scores_test[label] = pd.DataFrame(columns=df_columns)
            else:
                self.scores[label] = {}
                self.scores_test[label] = {}
                # print label_list
                for k in label_list:
                    self.scores[label][k] = pd.DataFrame(columns=df_columns)
                    self.scores_test[label][k] = pd.DataFrame(columns=df_columns)
                self.scores[label]['micro'] = pd.DataFrame(columns=df_columns)
                self.scores[label]['macro'] = pd.DataFrame(columns=df_columns)
                self.scores_test[label]['micro'] = pd.DataFrame(columns=df_columns)
                self.scores_test[label]['macro'] = pd.DataFrame(columns=df_columns)


    # i. SETUP #
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

    def _build_crossvalidation_folds(self, y):
        self.fold_features = defaultdict(dict)
        self.fold_features_test = defaultdict(dict)
        self.folds = defaultdict(dict)
        self.fold_probabilities = defaultdict(dict)
        self.fold_probabilities_test = defaultdict(dict)
        self.fold_predictions = defaultdict(dict)
        self.fold_predictions_test = defaultdict(dict)

        self.models = defaultdict(dict)
        self.scalers = defaultdict(dict)
        self.reducers = defaultdict(dict)

        sss = StratifiedShuffleSplit(y=y['classification'],
                n_iter=self.nfolds,
                test_size=self.fold_size,
                random_state=1)
        for i, (train_index, test_index) in enumerate(sss):
            self.folds[i] = {'train': train_index, 'test': test_index}
            # fold_features[fold][result_type][time] => np_array of features
            self.fold_features[i] = {
                'detection':defaultdict(dict),
                'gram':defaultdict(dict),
                'classification': defaultdict(dict)}
            self.fold_features_test[i] = {
                'detection':defaultdict(dict),
                'gram':defaultdict(dict),
                'classification': defaultdict(dict)}
            self.scalers[i] = {
                'detection':defaultdict(dict),
                'gram':defaultdict(dict),
                'classification': defaultdict(dict)}
            self.reducers[i] = {
                'detection':defaultdict(dict),
                'gram':defaultdict(dict),
                'classification': defaultdict(dict)}
            self.models[i] = {
                'detection':defaultdict(dict),
                'gram':defaultdict(dict),
                'classification': defaultdict(dict)}

            self.fold_probabilities[i] = {}
            self.fold_probabilities_test[i] = {}
            for k, v in self.confusion_labels.iteritems():
                self.fold_probabilities[i][k] = np.zeros((len(train_index), len(v)))
                self.fold_probabilities_test[i][k] = np.zeros((len(test_index), len(v)))
                self.fold_predictions[i][k] = np.zeros(len(train_index))
                self.fold_predictions_test[i][k] = np.zeros(len(test_index))


    ### UTILITY METHODS ###
    def _subset_data(self, Z, number_of_times):
        z_sub = Z.copy()
        z_sub= z_sub.apply(lambda x: x[0:number_of_times])
        return z_sub

    def _populate_score_dict(self, cm, results, number_of_times):
        score_dict = {}
        score_dict['time'] = number_of_times
        score_dict['confusion_matrix'] = cm
        score_dict['accuracy'] = results[0]
        score_dict['precision'] = results[1]
        score_dict['recall'] = results[2]
        score_dict['f1'] = results[3]

        return score_dict
    def _get_fold(self, X, fold, time):
        X_detection_train = X[fold]['detection'][train]
        X_detection_test = X[fold]['detection'][test]
        X_gram_train = X[fold]['gram'][train]
        X_gram_test = X[fold]['gram'][test]
        X_classification_train = X[fold]['classification'][train]
        X_classification_test = X[fold]['classification'][test]

        return (X_detection_train, X_detection_test, X_gram_train, X_gram_test, X_classification_train, X_classification_test)

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

    def _get_column_value_by_time(self, df, column, time):
        return df[df['time']==time][column].values[0]

    def _append_row_to_df(self, df, row):
        df.loc[len(df)+1] = row

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

    # 0) PREPROCESS #
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

    # 0) PREPROCESS #
    def _calculate_differences(self, x, reference_time):
        z = np.copy(x)
        # DI(0<=reference_time) = 0
        # DI(t>reference_time) = I(t) - I(reference_time)
        z[reference_time:, 1:] = z[reference_time:, 1:] - \
                                    z[reference_time, 1:]
        z[:reference_time, 1:] = 0
        return z

    # 0) PREPROCESS #
    def _rgb_to_hsv(self, X):
        return X

    # Workhorse helper methods #
    def _subset_fold_y_class(self, y, fold, test_train, result_type):
        y_test_train = y.iloc[self.folds[fold][test_train]][result_type].values
        return y_test_train

    def _subset_fold_y_testtrain(self, y, fold, test_train):
        yd = self._subset_fold_y_class(y, fold, test_train, result_type='detection')
        yg = self._subset_fold_y_class(y, fold, test_train, result_type='gram')
        yc = self._subset_fold_y_class(y, fold, test_train, result_type='classification')

        return (yd, yg, yc)

    def _subset_fold_y(self, y, fold):
        ytest = self._subset_fold_y_testtrain(y, fold, test_train = 'test')
        ytrain = self._subset_fold_y_testtrain(y, fold, test_train = 'train')

        return (ytrain, ytest)

    def _subset_fold_X_class(self, fold, test_train, result_type, t):
        # print fold, test_train, result_type, t
        if test_train == 'train':
            X_test_train = self.fold_features[fold][result_type][t]
        else:
            X_test_train = self.fold_features_test[fold][result_type][t]
        return X_test_train

    def _subset_fold_X_testtrain(self, fold, test_train, t):
        # print fold, test_train, t
        Xd = self._subset_fold_X_class(fold, test_train, result_type = 'detection', t=t)
        Xg = self._subset_fold_X_class(fold, test_train, result_type = 'gram', t=t)
        Xc = self._subset_fold_X_class(fold, test_train, result_type = 'classification', t=t)

        return (Xd, Xg, Xc)

    def _subset_fold_X(self, fold, t):
        X_train = self._subset_fold_X_testtrain(fold, test_train = 'train', t=t)
        X_test = self._subset_fold_X_testtrain(fold, test_train = 'test', t=t)

        return (X_train, X_test)

    def _accumulate_results(self, result_dict, results):
        (rd, rg, rc) = results

        result_dict['detection'].extend(rd)
        result_dict['gram'].extend(rg)
        result_dict['classification'].extend(rc)


    ### DEPRECATED METHODS ###
    def _prepare_data(self, X):
        Z = self.preprocess(X)

        self._build_results_dataframes(len(Z))
        self.trial_lengths = self.find_trial_lengths(Z)
        self.inspect_trial_shapes(Z)

        return Z

    ### NOT COMPLETED METHODS ###

    # def __repr__(self):
    #     print self

    def bayes_update(self,t):
        # use Bayesian prior/posterior ideas to update predictions, probabilities
        # based on most recent timestep
        pass




    #### WORKHORSE METHOD ####
    def fit(self, X, y, verbose=False, debug=False):
        self.verbose = verbose
        self.debug = debug
        # start with the second time
        t = self.reference_time + self.min_time
        # don't use results for the first model made
        use_last_timestep_results = False
        if debug:
            self.times = [30,40,50]
        else:
            self.times = np.arange(t, self.max_time, 1)

        # i) SETUP # GOOD
        self.setup(X,y)

        # Check trial integrity
        if self.max_time > self.trial_lengths.min():
            ptf( '***FIT ERROR***', self.logfile)
            ptf( 'Minimum trial_length, %s, is less than max_time, %s' % (self.trial_lengths.min(), self.max_time), self.logfile)
            return

        # generate all features or load all features
        if self.featurizer_coldstart:
            # 0) PREPROCESS All trials
            X_preprocessed = self.preprocess(X)
            # 1) FEATURIZE - step by timestep
            X_featurized = self.featurize(X_preprocessed, self.featurizer_pickle)

            # 1A) PICKLE FEATURES #
            start = time.time()
            ptf('\n>> 1A. Pickling features to %s ...' % self.features_pickle, self.logfile)
            self.pickle_features(X_featurized, self.features_pickle)
            end = time.time()
            ptf('\n>> Pickling completed (%s seconds) <<' % (end-start), self.logfile)

        else:
            start = time.time()
            ptf('\n>> 1. Loading features from %s ...' % self.features_pickle, self.logfile)

            X_featurized = self.load_features(self.features_pickle)
            self.features = X_featurized
            end = time.time()
            ptf('\n>> Loading completed (%s seconds) <<' % (end-start), self.logfile)


        if self.scaler_coldstart or self.reducer_coldstart:
            # 2) SCALE - step by timestep and by fold
            X_scaled = self.scale(X_featurized, self.scaler_pickle)
            # 3) REDUCE - step by timestep and by fold
            X_reduced = self.reduce(X_scaled, self.reducer_pickle)

            start = time.time()
            ptf('\n>> 4A. Pickling final features to %s, %s' % (self.fold_features_pickle, self.fold_features_test_pickle), self.logfile)
            self.pickle_features(X_reduced, self.fold_features_pickle)
            self.pickle_features(self.fold_features_test, self.fold_features_test_pickle)
            ptf('\n>> Pickling completed (%s seconds) <<' % (end-start), self.logfile)

        else:
            ptf('\n>> 1-4A. Loading fold_features from %s ...' % self.fold_features_pickle, self.logfile)
            X = self.load_features(self.fold_features_pickle)
            self.fold_features = X
            Xt = self.load_features(self.fold_features_test_pickle)
            self.fold_features_test = Xt

        # Now we fit
        ptf('\n\n>> 5. Training, 6. Predicting, 7. Scoring and 8. Storing ... \n\n', self.logfile)
        fitstart = time.time()

        use_last_timestep_results = False # don't use for first timestep
        for t in self.times:
            start = time.time()

            if self.verbose:
                ptf( '\n\nTIMESTEP %d...' % t, self.logfile)

            y_train_true_timestep = defaultdict(list)
            y_train_predict_timestep = defaultdict(list)
            y_test_true_timestep = defaultdict(list)
            y_test_predict_timestep = defaultdict(list)

            results_dicts = (
                y_train_true_timestep,
                y_train_predict_timestep,
                y_test_true_timestep,
                y_test_predict_timestep)

            for fold, fold_indexes in self.folds.iteritems():
                # get Xd, Xg, Xc for this fold and timestep
                (X_train, X_test) = self._subset_fold_X(fold, t)

                # X_train_detection = self.fold_features[fold]['detection'][t]
                # X_train_gram = self.fold_features[fold]['gram'][t]
                # X_train_classification = self.fold_features[fold]['classification'][t]

                # get yd, yg, yc for this fold and timestep
                (y_train, y_test) = self._subset_fold_y(y, fold)

                # y_train_detection = self._subset_fold_y(y, fold, 'detection')
                # y_train_gram = self._subset_fold_y(y, fold, 'gram')
                # y_train_classification = self._subset_fold_y(y, fold, 'classification')

                # X = [X_train_detection, X_train_gram, X_train_classification]
                # y = [y_train_detection, y_train_gram, y_train_classification]

                # train on Xtrain fold features
                # 5) TRAIN
                models, train_predictions, train_probabilities = self.train(X_train, y_train, fold, t, use_last_timestep_results)
                # 6) PREDICT
                test_predictions, test_probabilities = self.predict(models, X_test, fold, t, use_last_timestep_results)

                # (model_detection, model_gram, model_classification) = models
                # (y_train_predictions_detection, y_train_predictions_gram, y_train_predictions_classification) = train_predictions
                # (y_train_probabilities_detection, y_train_probabilities_gram, y_train_probabilities_classification) = train_probabilities

                # accumulate y_pred_train, y_true_train
                results = (
                    y_train,
                    train_predictions,
                    y_test,
                    test_predictions)
                for result_dict, result in izip(results_dicts, results):
                    self._accumulate_results(result_dict, result)

                # 7) STORE
                self._store_one_fold(
                    (train_predictions, train_probabilities),
                    (test_predictions, test_probabilities),
                    fold,
                    t)

            # 8) SCORE
            self._score_one_timestep(results_dicts, t)
            use_last_timestep_results = self.use_last_timestep_results

            # Bayes update
            # if not self.use_last_timestep_results:
            #     self.bayes_update(t)

            end = time.time()
            if self.verbose:
                ptf( '\n----TIMESTEP %d took %d seconds----\n\n' % (t, (end-start)), self.logfile)
        fitend = time.time()
        ptf('\n\n>> 5. Training, 6. Predicting, 7. Scoring and 8. Storing completed <<', self.logfile)
        ptf('%d seconds (%0.1f mins) for:\tTrials:%d, Folds:%d, Times:%d\n\n' %
                ((fitend-fitstart),
                (fitend-fitstart)/60.0,
                len(X),
                self.nfolds,
                len(self.times)),
            self.logfile)




if __name__ == '__main__':
    print 'Series Model imported'
