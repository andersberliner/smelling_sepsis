'''
tsm.py
Anders Berliner
20160105

Contains the series model class.

'''
from seriesmodel import SeriesModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedShuffleSplit
from itertools import izip
from collections import defaultdict
from featurizer import PolynomialFeaturizer, KineticsFeaturizer, DerivativeFeaturizer
import multiclassmetrics as mcm
import time
from functools import partial
import pickle
import os
from nonetype import NoneTypeScaler, NoneTypeReducer, NoneTypeFeaturizer

from output_capstone import print_to_file_and_terminal as ptf

import utils_seriesmodel as usm

class TriggeredSeriesModel(SeriesModel):
    def __init__(self,
            column_headers,
            # run conditions
            verbose = False,
            logfile = None,
            on_disk = True, # whether to keep results in memory or write to disk after each timestep
            runid = 'output',
            load_state = None,
            load_time = 0,
            max_time = 60, # 20 hrs max
            min_time = 1, # time after ref_time to start fitting
            max_postdetection_time = 27, # 9 hours after detection
            # crossvalidation options
            nfolds=1,
            fold_size=0.1,
            # i) Preprocessing conditions #
            color_scale = 'RGB',
            color_vector_type = 'DII',
            reference_time = 12,
            # A) FIT conditions #
            # A1) TRIGGER conditions #
            trigger_threshold = 0.5,
            trigger_pickle = 'triggers.pickle',
            trigger_spots = ['26B', '11R', '45B', '36B', '30R', '11B'],
            detection_featurizer_arguments = {'order':2, 'dx':20.0, 'maxmin':True},
            detection_featurizer = 'derivative',
            # Pickles
            featurizer_pickle = 'featurizer.pkl',
            features_pickle = 'features.pkl',
            fold_features_pickle = 'fold_features.pkl',
            fold_features_test_pickle = 'fold_features_test.pkl,
            # Deprecated conditions required by parent class
            gram_featurizer = None,
            gram_featurizer_arguments = None,
            classification_featurizer = None,
            classification_featurizer_arguments = None,
            # B TRAIN conditions #
            use_last_timestep_results = False, # True for simple, False do to pseuduo-bayesian approach
            classification_spots = None,
            last=1):
        # run conditions
        self.logfile = logfile
        self.on_disk = on_disk
        self.runid = runid
        self.max_time = max_time
        self.min_time = min_time
        self.load_state = load_state
        self.load_time = load_time
        # used by parent class for recylced methods
        self.trigger = True

        # crossvalidation conditions
        self.nfolds = nfolds
        self.fold_size = fold_size

        # i) Preprocessing conditions #
        self.color_scale = color_scale
        self.color_vector_type = color_vector_type
        self.reference_time = reference_time
        self.number_of_columns = 220 # expected number of spots and colors + time
        self.column_headers = column_headers
        # subset of spots to consider for gram and classification
        # if None, use all the spots passed-in
        self.classification_spots = classification_spots

        # A) FIT conditions #
        # A1) TRIGGER conditions #
        self.trigger_threshold = trigger_threshold
        self.trigger_pickle = trigger_pickle
        self.trigger_spots = trigger_spots

        self.detection_base_featurizer_arguments = detection_featurizer_arguments
        self.detection_base_featurizer = detection_featurizer
        # Pickles
        self.featurizer_pickle = featurizer_pickle
        self.features_pickle = features_pickle
        self.fold_features_pickle = fold_features_pickle
        self.fold_features_test_pickle = fold_features_test_pickle
        # Deprecated conditions required by parent class
        self.gram_base_featurizer = gram_featurizer
        self.gram_base_featurizer_arguments = gram_featurizer_arguments
        self.classification_base_featurizer = classification_featurizer
        self.classification_base_featurizer_arguments = classification_featurizer_arguments


        # B) TRAIN conditions #
        self.use_last_timestep_results = use_last_timestep_results

        # other
        self.stages = ['start', 'preprocess', 'featurize']

    ### MAIN METHODS ###
    # use base class repr

    # i) SETUP #
    def setup(self, X, y):
        '''
        Inspects data to build crossvalidation folds, initialize results lists
        and dicts, and build other helper lists and dicts

        IN:
            SeriesModel
            X - pd dataframe - trial data.  see data structures.  Usually passed in to fit
            y - pd dataframe - trial labels.  see data structures.  Usually passed in to fit
        OUT: None
        '''
        start = time.time()
        ptf('\n>> i. Setting-up TriggeredSeriesModel ...', self.logfile)
        # Use parent class methods
        self.confusion_labels = self._build_confusion_labels(y)
        self.trial_lengths = self.find_trial_lengths(X)
        self.inspect_trial_shapes(X)
        self._build_results_dataframes(len(X))

        # define new methods
        self._build_crossvalidation_folds(y)


        end = time.time()
        ptf('\n>> Set-up completed (%s seconds) <<' % (end-start), self.logfile)

    # NOTE - overridden from parent class
    def _build_crossvalidation_folds(self, y):
        # detection, gram and classification
        self.fold_features = defaultdict(dict)
        self.fold_features_test = defaultdict(dict)
        self.folds = defaultdict(dict)
        self.fold_probabilities = defaultdict(dict)
        self.fold_probabilities_test = defaultdict(dict)
        self.fold_predictions = defaultdict(dict)
        self.fold_predictions_test = defaultdict(dict)

        # gram and classification
        self.models = defaultdict(dict)
        self.scalers = defaultdict(dict)
        self.reducers = defaultdict(dict)

        # TSM
        # shifted time axis for each fold, trial
        self.tau = defaultdict(None)
        self.tau_test = defaultdict(None)
        # trigger time for each fold, trial
        self.trigger = defaultdict(None)
        self.triggered = defaultdict(None)
        self.trigger_test = defaultdict(None)
        self.triggered_test = defaultdict(None)
        # trigger features for each time
        self.trigger_features = defaultdict(None)
        self.trigger_times = defaultdict(None)
        # detection results for each fold, timestep
        self.trigger_results = defaultdict(dict)
        self.trigger_results_test = defaultdict(dict)

        # create cross validation folds
        sss = StratifiedShuffleSplit(y=y['classification'],
                n_iter=self.nfolds,
                test_size=self.fold_size,
                random_state=1)

        for i, (train_index, test_index) in enumerate(sss):
            self.folds[i] = {'train': train_index, 'test': test_index}

            # TSM
            self.trigger_results[i] = pd.DataFrame(columns=['time', 'trigger_times',
                'trigger_values', 'probabilities', 'tprs', 'fprs', 'thresholds'])
            # for each fold, trigger time for each trial, be it train or test
            self.trigger[i] = np.zeros(len(y))
            self.triggered[i] = np.zeros(len(y))
            self.tau[i] = defaultdict(None)
            for trial in range(0, len(y)):
                self.tau[i][trial] = np.zeros((self.max_time, 1))

            self.trigger_test[i] = np.zeros(len(y))
            self.triggered_test[i] = np.zeros(len(y))
            self.tau_test[i] = defaultdict(None)
            #
            # self.trigger_features[i] = defaultdict(None)
            # self.trigger_features_test[i] = defaultdict(None)
            for trial in range(0, len(y)):
                self.tau_test[i][trial] = np.zeros((self.max_time, 1))

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


        # 0A) PRUNE #
        def prune_spots(self, X, trigger_spots, column_headers):
            '''
            Prunes preprocdessed data to just spots in trigger_spots
            IN:
                SeriesModel
                X - pd dataframe - preprocessed trial data.  see data structures.  Usually passed in to fit
                trigger_spots - list - list of str names of spots to keep.
                column_headers - list - list of str names of all columns
            OUT:
                X - pd dataframe - preprocessed trial data
            '''
            if self.beyond('preprocess'):
                ptf('\n>> 0. Skipped Preprocessing << \n', self.logfile)
                return None
            elif self.load_state == 'preprocess':
                ptf('\n>> 0. LOADING Preprocessed data ...', self.logfile)
                X = self.load_time_step('DI')
                return X
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
            if self.on_disk:
                self.pickle_time_step(X, 'DI')
            return X

    # 1) FEATURIZE #
    def featurize_triggers(self, X, featurizer_pickle, fold, t):
        '''
        Extracts features for detection, gram, classification from preprocessed
        data using conditions passed to init.
        IN:
            SeriesModel
            X - pd dataframe - preprocessed trial data
            featurizer_pickle - str - file name to store featurizers
                NOTE: if on_disk = True, pickles are stored at each timestep
                using runid as filename prefix.
        OUT:
            X - dict of dict of np_arrays - scaled features (ntrials X nfeatures) for each
                timestep (second key), and class(first key)
        '''
        start = time.time()
        number_of_times = t
        # featurize, storing featurizers at each timestep
        if self.verbose:
            ptf( '> 1. Featurizing nt=%d ...' % number_of_times, self.logfile)

        X_train = self._subset_data(X, number_of_times)
        if self.debug:
            print t, X.iloc[0].shape

        X_trigger, trigger_times = self._featurize_class(X_train,
            self.trigger_base_model, self.trigger_base_model_arguments)
        if self.debug:
            print t, X_detection.iloc[0].shape, trigger_times.iloc[0].shape

        # convert to numpy arrays
        np_X_trigger = self._pandas_to_numpy(X_trigger)
        np_trigger_times = self._pandas_to_numpy(trigger_times)
        if self.debug:
            print 'Checking featurized shapes', np_X_trigger.shape, np_trigger_times.shape

        # store features
        if not self.on_disk:
            self.trigger_features[t] = np_X_trigger
            self.trigger_times[t] = np_trigger_times
        else:
            self.pickle_time_step(np_X_trigger, 'trigger_features', t)
            self.pickle_time_step(np_X_trigger, 'trigger_times', t)

        end = time.time()
        ptf('\n...(%s seconds) <' % (end-start), self.logfile)

        # pickle featurizers
        if not self.on_disk:
            featurizer_file_name = featurizer_pickle
            start = time.time()
            ptf('\n>> Pickling featurizers to %s ...' % featurizer_file_name, self.logfile)

            self.pickle_time_step(self.featurizers, 'featurizer', file_name = featurizer_file_name)

            end = time.time()
            ptf('\n>> Pickling completed (%s seconds) <<' % (end-start), self.logfile)

        return self.features

    ### WORKHORSE methods ###
    # A) FIT
    def fit(self, X, y, verbose=False, trigger_only=True, debug=False):
        self.trigger_only = trigger_only
        self.verbose = verbose
        self.debug = debug

        # start with the second time
        tmin = self.reference_time + self.min_time
        if debug:
            self.times = [30,40,50]
        else:
            self.times = np.arange(tmin, self.max_time, 1)

        # i) SETUP #
        self.setup(X,y)

        # Check trial integrity
        self._check_trial_integrity()

        # 0) PREPROCESS All trials
        X_preprocessed = self.preprocess(X)

        # check load state and run only needed times
        if self.load_state == 'featurize':
            ptf('\n>> 1. Computing triggers from timestep %d ...' % self.load_time, self.logfile)
            run_times = self.make_run_times(self.times, self.load_time)
        else:
            ptf('\n>> 1. Computing triggers from first timestep %d ...' % tmin, self.logfile)
            run_times = self.times

        for t in run_times:
            start = time.time()
            if self.verbose:
                ptf('\n\nTIMESTEP %d...' %t, self.logfile)
            # 1) trigger_featurize
            X_featurized = self.featurize_triggers(X_preprocessed, self.featurizer_pickle)

            # 2) trigger_train

            # 3) trigger_predict

            # 4) trigger_score

            if self.trigger_only:
                return

        # 4A) Write avg tau to file for each trial.  We will pass this into triggered series model

        ## UPDATES FOR V2 BEYOND THIS ##

        # 5) tranform tau

        # for t in range(0, self.max_postdetection_time, 20):
