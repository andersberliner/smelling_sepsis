'''
tsm.py
Anders Berliner
20160105

Contains the series model class.

'''
from seriesmodel import SeriesModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc
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
# https://github.com/glemaitre/UnbalancedDataset/blob/master/notebook/Notebook_UnbalancedDataset.ipynb
from unbalanced_dataset import UnderSampler, SMOTE
from output_capstone import print_to_file_and_terminal as ptf

import utils_seriesmodel as usm

class TriggeredSeriesModel(SeriesModel):
    def __init__(self,
            column_headers,
            # run conditions
            debug = False,
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
            nfolds=10,
            fold_size=0.1,
            # i) Preprocessing conditions #
            color_scale = 'RGB',
            color_vector_type = 'DII',
            reference_time = 12,
            # A) FIT conditions #
            resample_method = None, # [None, 'under', 'over']
            undersample_method = 'random',
            undersample_arguments = {'ratio':0.1},
            oversample_method = 'SMOTE',
            oversample_arguments = {'ratio':10.0, 'kind':'regular'},
            # A1) TRIGGER conditions #
            trigger_threshold = 0.5,
            trigger_pickle = 'triggers.pickle',
            trigger_spots = ['26B', '11R', '45B', '36B', '30R', '11B'],
            detection_featurizer_arguments = {},
            detection_featurizer = 'derivative',
            detection_model = 'LRCV',
            detection_model_arguments = {},
            detection_scaler='SS',
            detection_scaler_arguments={},
            detection_reducer='pca',
            detection_reducer_arguments={},
            # Pickles
            featurizer_coldstart = True,
            featurizer_pickle = 'featurizer.pkl',
            features_pickle = 'features.pkl',
            fold_features_pickle = 'fold_features.pkl',
            fold_features_test_pickle = 'fold_features_test.pkl',
            # Deprecated conditions required by parent class
            reducer_pickle = 'reducer.pkl',
            reducer_coldstart = True,
            scaler_pickle = 'scaler.pkl',
            scaler_coldstart = True,
            gram_model='LR',
            gram_model_arguments={},
            gram_reducer='detection',
            gram_reducer_arguments={},
            gram_featurizer='detection',
            gram_featurizer_arguments={},
            gram_scaler='detection',
            gram_scaler_arguments={},
            classification_model='LR',
            classification_model_arguments={},
            classification_reducer='detection',
            classification_reducer_arguments={},
            classification_featurizer='gram',
            classification_featurizer_arguments={},
            classification_scaler='gram',
            classification_scaler_arguments={},
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
        self.verbose = verbose
        self.debug = debug
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
        # resampling CONDITIONS
        self.resample_method = resample_method
        self.undersample_method = undersample_method
        self.undersample_arguments = undersample_arguments
        self.oversample_method = oversample_method
        self.oversample_arguments = oversample_arguments

        # A1) TRIGGER conditions #
        self.trigger_threshold = trigger_threshold
        self.trigger_pickle = trigger_pickle
        self.trigger_spots = trigger_spots

        self.detection_base_model = detection_model
        self.detection_base_reducer = detection_reducer
        self.detection_base_featurizer = detection_featurizer
        self.detection_base_scaler = detection_scaler

        self.detection_base_model_arguments = detection_model_arguments
        self.detection_base_reducer_arguments = detection_reducer_arguments
        self.detection_base_featurizer_arguments = detection_featurizer_arguments
        self.detection_base_scaler_arguments = detection_scaler_arguments

        # Pickles
        self.featurizer_pickle = featurizer_pickle
        self.features_pickle = features_pickle
        self.fold_features_pickle = fold_features_pickle
        self.fold_features_test_pickle = fold_features_test_pickle
        self.featurizer_coldstart = featurizer_coldstart
        # Deprecated conditions required by parent class
        self.reducer_pickle = reducer_pickle
        self.reducer_coldstart = reducer_coldstart
        self.scaler_pickle = scaler_pickle
        self.scaler_coldstart = scaler_coldstart

        self.gram_base_model = gram_model
        self.gram_base_reducer = gram_reducer
        self.gram_base_featurizer = gram_featurizer
        self.gram_base_scaler = gram_scaler

        self.classification_base_model = classification_model
        self.classification_base_reducer = classification_reducer
        self.classification_base_featurizer = classification_featurizer
        self.classification_base_scaler = classification_scaler

        self.gram_base_model_arguments = gram_model_arguments
        self.gram_base_reducer_arguments = gram_reducer_arguments
        self.gram_base_featurizer_arguments = gram_featurizer_arguments
        self.gram_base_scaler_arguments = gram_scaler_arguments

        self.classification_base_model_arguments = classification_model_arguments
        self.classification_base_reducer_arguments = classification_reducer_arguments
        self.classification_base_featurizer_arguments = classification_featurizer_arguments
        self.classification_base_scaler_arguments = classification_scaler_arguments


        # B) TRAIN conditions #
        self.use_last_timestep_results = use_last_timestep_results

        # other
        self.stages = ['start', 'preprocess', 'prune', 'featurize']

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
        self.trigger_times = defaultdict(None)
        self.triggered = defaultdict(None)
        self.trigger_test = defaultdict(None)
        self.triggered_test = defaultdict(None)
        # trigger features for each time
        self.trigger_features = defaultdict(None)
        self.trigger_feature_times = defaultdict(None)
        self.trigger_featurizers = defaultdict(None)
        # detection results for each fold, timestep
        self.trigger_results = defaultdict(dict)
        self.trigger_results_test = defaultdict(dict)
        # trigger models for each fold, timestep
        self.trigger_models = defaultdict(dict)

        # resampling labels for each timestep
        self.trigger_resample_labels = defaultdict(dict)
        self.trigger_resample_features = defaultdict(dict)

        # TSM
        cols = ['time', 'fold']
        for col in ['predictions', 'probabilities']:
            for testtrain in ['test', 'train']:
                cols.append(testtrain + '_' + col)
        self.trigger_results = pd.DataFrame(columns=cols)

        df_columns = ['time', 'fold']
        self.metrics = ['confusion_matrix', 'accuracy', 'precision', 'recall',
            'f1', 'overall_accuracy', 'fpr', 'tpr', 'thresholds', 'auc']
        df_columns.extend(self.metrics)
        self.trigger_scores = pd.DataFrame(columns=df_columns)
        self.trigger_scores_test = pd.DataFrame(columns=df_columns)

        # create cross validation folds
        sss = StratifiedShuffleSplit(y=y['classification'],
                n_iter=self.nfolds,
                test_size=self.fold_size,
                random_state=1)

        # print sss.n_iter, self.nfolds, len(sss)

        for i, (train_index, test_index) in enumerate(sss):
            print i, len(test_index), len(train_index)
            self.folds[i] = {'train': train_index, 'test': test_index}

            # at each timestep we will append a df with labels
            self.trigger_resample_labels[i] = defaultdict(None)
            self.trigger_resample_features[i] = defaultdict(None)
            # for each fold, trigger time for each trial, be it train or test
            self.trigger_times[i] = np.zeros(len(y))
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




    # 1) FEATURIZE #
    def featurize_triggers(self, X, t):
        '''
        Extracts features for detection, gram, classification from pruneed
        data using conditions passed to init.
        IN:
            SeriesModel
            X - pd dataframe - preprocessed trial data
            t - int - time index
        OUT:
            X - np_array - extracted features (ntrials X nfeatures) as this timestep
        '''
        start = time.time()
        number_of_times = t
        # featurize, storing featurizers at each timestep
        if self.verbose:
            ptf( '> 1. Featurizing nt=%d ...' % number_of_times, self.logfile)

        X_train = self._subset_data(X, number_of_times)
        if self.debug:
            print t, X.iloc[0].shape

        (X_trigger, trigger_times), trigger_featurizer = self._featurize_class(X_train,
            self.detection_base_featurizer, self.detection_base_featurizer_arguments)
        if self.debug:
            print t, X_trigger.iloc[0].shape, trigger_times.iloc[0].shape

        # convert to numpy arrays
        np_X_trigger = self._pandas_to_numpy(X_trigger)
        np_trigger_times = self._pandas_to_numpy(trigger_times)
        if self.debug:
            print 'Checking featurized shapes', np_X_trigger.shape, np_trigger_times.shape

        # store features
        if not self.on_disk:
            self.trigger_features[t] = np_X_trigger
            self.trigger_feature_times[t] = np_trigger_times
            self.trigger_featurizers[t] = trigger_featurizer
        else:
            self.pickle_time_step(np_X_trigger, 'trigger_features', t)
            self.pickle_time_step(np_trigger_times, 'trigger_feature_times', t)
            self.pickle_time_step(trigger_featurizer, 'trigger_featurizer', t)

        # Append results to results df later after scoring
        end = time.time()
        ptf('\n...(%s seconds) <' % (end-start), self.logfile)

        return np_X_trigger

    def _subset_fold_triggers(self, X, fold):
        '''
        Extracts the featurized (as triggers) Xdata for a given fold from X

        IN
            X - np array - ntrials X ntriggers array of triggers for a given timestep
            fold - int - index of the fold
        OUT
            Xsub - np array - ntrials_in_fold X ntriggers
        '''
        Xtest = X[self.folds[fold]['test']]
        Xtrain = X[self.folds[fold]['train']]

        return Xtrain, Xtest

    def trigger_train(self, X, y, fold, t):
        '''
        Trains models for a timestep, fold
        IN:
            TriggeredSeriesModel
            X - nparrays - final features for this train fold, timestep (ntrials X nfeatures)
            y - dict of nparrays - labels for this train fold (ntrials) for each label class (key)
            fold - int - fold index
                NOTE: only the detection label is used in this implementation, but all are passed
                in for backwords compatibility with series model
            t - int - time index
        OUT:
            models - model - trained model for this timestep, fold
            predictions - np array - train predictions nparray (ntrials in this fold)
            probabilities - np array - train probabilities nparray (ntrials in this fold X 2)
        '''
        number_of_times = t
        # (X_train_detection, X_train_gram, X_train_classification) = X
        np_X_detection = X
        print len(y)
        (y_train_detection, y_train_gram, y_train_classification) = y

        # fit detection
        if self.verbose:
            ptf( 'Training detection fold:%d, nt:%d ...' % (fold, number_of_times), self.logfile)

        model_detection = self._fit_class(np_X_detection,
            y_train_detection,
            self.detection_base_model,
            self.detection_base_model_arguments,
            step=('detection t=%d_%d' % (fold,number_of_times)))

        # store model, predict
        y_predict_detection = model_detection.predict(np_X_detection)
        y_probabilities_detection = model_detection.predict_proba(np_X_detection)

        if not self.on_disk:
            self.trigger_models[fold][number_of_times] = model_detection

        else:
            self.pickle_time_step(model_detection, 'trigger_model', t=number_of_times, fold=fold)

        return model_detection, y_predict_detection, y_probabilities_detection

    def _trigger_store_one_fold(self, train, test, fold, t):
        (train_predictions, train_probabilities) = train
        (test_predictions, test_probabilities) = test
        results_dict = {}
        results_dict['time'] = t
        results_dict['fold'] = fold
        for (predictions, probabilities), testtrain_type in izip((test, train), ['test', 'train']):
            # print testtrain_type, predictions.shape, probabilities.shape
            results_dict[testtrain_type + '_predictions'] = predictions
            results_dict[testtrain_type + '_probabilities'] = probabilities

        self._append_row_to_df(self.trigger_results, results_dict)

        # update fold predictions and probabilities
        # COME BACK TO THIS ?? #

    def trigger_predict(self, model_detection, X_test, fold, t):
        number_of_times = t
        if self.verbose:
            ptf( 'Predicting detection fold:%d, nt:%d ...' % (fold, number_of_times), self.logfile)

        y_predict_detection = model_detection.predict(X_test)
        y_probabilities_detection = model_detection.predict_proba(X_test)

        return y_predict_detection, y_probabilities_detection

    def _trigger_populate_score_dict(self, cm, results, number_of_times,
            fpr, tpr, thresholds, roc_auc, overall_acc='NA'):
        score_dict = {}
        score_dict['time'] = number_of_times
        score_dict['confusion_matrix'] = cm
        score_dict['accuracy'] = results[0]
        score_dict['precision'] = results[1]
        score_dict['recall'] = results[2]
        score_dict['f1'] = results[3]
        score_dict['fpr'] = fpr
        score_dict['tpr'] = tpr
        score_dict['thresholds'] = thresholds
        score_dict['auc'] = roc_auc
        # RETURN TO THIS #
        score_dict['overall_accuracy'] = overall_acc

        return score_dict

    def _trigger_score_one_fold(self, yt, yp, probas, t, testtrain='test', fold='all'):
        number_of_times = t
        fpr, tpr, thresholds = roc_curve(yt, probas[:,1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        if self.verbose and fold=='all':
            ptf('%s results' % testtrain, self.logfile)
            ptf(mcm.classification_report_ovr(yt, yp, self.confusion_labels['detection']), self.logfile)

        scores = mcm.scores_binary(yt, yp)
        # builds confusion matrix of TP, FP, etc. for the detection case
        cm = mcm.confusion_matrix_binary(yt, yp)
        # detection - populate scores
        overall_acc = accuracy_score(yt, yp)
        score_dict = self._trigger_populate_score_dict(cm, scores, number_of_times,
            fpr, tpr, thresholds, roc_auc, overall_acc=overall_acc)

        score_dict['fold'] = fold
        if testtrain == 'train':
            self._append_row_to_df(self.trigger_scores, score_dict)
        else:
            self._append_row_to_df(self.trigger_scores_test, score_dict)

    def find_new_folds(self, Xsmote, ysmote, y):
        # the first n elements of y are unchanged in ysmote
        # need to make newdf with additional values
        # then build folds of classification based on that

        ysmotedf = pd.DataFrame({
            'detection': smoy[len(y):],
            'classification': ['Control']*(len(smoy[len(y):])),
            'gram': ['Control']*(len(smoy[len(y):])),
            'synthetic': [1]*(len(smoy[len(y):]))})

        newydf = pd.concat([y, sdf])
        newydf = newydf.dropna(0)

        sss = StratifiedShuffleSplit(y=newydf['classification'],
                n_iter=self.nfolds,
                test_size=self.fold_size,
                random_state=1)
        folds = defaultdict(list)
        for i, (train_index, test_index) in enumerate(sss):
            folds[i] = {'train': train_index, 'test': test_index}

        return folds


    def resample(self, X, y, t, fold):
        if not self.resample_method:
            return X, y
        else:
            start = time.time()
            if self.verbose:
                ptf('> Resampling for timestep %d, fold %d' % (t, fold), self.logfile)

            # create resampler
            if self.resample_method == 'under':
                print 'UNDER SAMPLING is not implemented yet'
                return X, y
            elif self.resample_method == 'over':
                if self.oversample_method.lower() == 'smote':
                    resampler = SMOTE(**self.oversample_arguments)
                else:
                    print 'Your resampling method is not implemented yet'
                    return X, y

            print type(X), type(y)
            print X.shape, y[0].shape
            Xsmote, ysmote = resampler.fit_transform(X, y[0])
            # resample
            ysmote_tuple = self.build_smoted_label_tuple(ysmote, y, fold)
            # ysmote_df = self.build_smoted_label_df(ysmote, y, fold)
            # # find new folds
            # folds, ynewdf = self.find_new_folds(Xsmote, ysmote, y)

            if self.debug:
                print np.sum(y[0]==0), np.sum(ysmote == 0)
                print np.sum(y[0]==1), np.sum(ysmote == 1)

            if self.on_disk:
                self.pickle_time_step(ysmote_df, 'trigger_resample_labels', fold=fold, t=t)
                self.pickle_time_step(Xsmote, 'trigger_resample_features', fold=fold, t=t)
            else:
                self.trigger_resample_labels[fold][t] = ysmote_tuple
                self.trigger_resample_features[fold][t] = Xsmote

            end = time.time()
            if self.verbose:
                ptf('... %d s' % (end-start), self.logfile)
            return Xsmote, ysmote_tuple

    # DEPRECATED
    def build_smoted_label_df(self, ysmote, y, fold):
        ysmoteddict = pd.DataFrame({
            'detection': ysmote[len(y):],
            'classification': ['Control']*(len(ysmote[len(y):])),
            'gram': ['Control']*(len(ysmote[len(y):])),
            'synthetic': [1]*(len(ysmote[len(y):]))})
        newydf = pd.concat([y, sdf])
        newydf = newydf.dropna(0)
        return newydf

    def build_smoted_label_tuple(self, ysmote, y, fold):
        detection = ysmote

        gram_list = y[1].tolist()
        gram_list = self.fill_list(gram_list, len(ysmote[len(y):]))
        gram = np.array(gram_list)

        classification_list = y[2].tolist()
        classification_list = self.fill_list(classification_list, len(ysmote[len(y):]))
        classification = np.array(classification_list)

        ysmoted_tuple = (detection, gram, classification)
        return ysmoted_tuple

    # DEPRECATED
    def build_smoted_label_dict(self, ysmote, y, fold):
        ysmoteddict = defaultdict(list)
        detection = ysmote

        gram_list = y['gram'].tolist()
        gram_list = self.fill_list(gram_list, len(ysmote[len(y):]))
        gram = np.array(gram_list)

        classification_list = y['classification'].tolist()
        classification_list = self.fill_list(classification_list, len(ysmote[len(y):]))
        classification = np.array(classification_list)

        ysmoteddict = {'detection': detection, 'gram': gram, 'classification': classification}
        return ysmoteddict

    def fill_list(self, x, extension_length, value='Control'):
        x.extend([value]*extension_length)
        return x

    ### WORKHORSE methods ###
    # A) FIT
    def fit(self, X, y, verbose=False, trigger_only=True, debug=False):
        self.trigger_only = trigger_only
        self.verbose = verbose
        self.debug = debug

        # start with the second time
        tmin = self.reference_time + self.min_time
        self.times = np.arange(tmin, self.max_time, 1)

        # i) SETUP #
        self.setup(X,y)

        # Check trial integrity
        self._check_trial_integrity()

        # 0) PREPROCESS All trials
        X_preprocessed = self.preprocess(X)
        X_pruned = self.prune_spots(X_preprocessed, self.trigger_spots, self.column_headers)

        # check load state and run only needed times
        if self.load_state == 'featurize':
            ptf('\n>> 1. Computing triggers from timestep %d ...' % self.load_time, self.logfile)
            run_times = self.make_run_times(self.times, self.load_time)
        else:
            ptf('\n>> 1. Computing triggers from first timestep %d ...' % tmin, self.logfile)
            run_times = self.times

        if debug:
            self.times = [30,40,50]
            run_times = self.times

        for t in run_times:
            start = time.time()
            if self.verbose:
                ptf('\n\nTIMESTEP %d...' %t, self.logfile)
            # 1) trigger_featurize
            X_featurized = self.featurize_triggers(X_pruned, t)

            # results to accumulate for this timestep
            y_train_true_timestep = []
            y_train_predict_timestep = []
            y_test_true_timestep = []
            y_test_predict_timestep = []
            y_train_probabilities = []
            y_test_probabilities = []

            for i, (fold, fold_indexes) in enumerate(self.folds.iteritems()):
                (X_train, X_test) = self._subset_fold_triggers(X_featurized, fold)
                (y_train, y_test) = self._subset_fold_y(y, fold)


                # resample
                X_resampled, y_resampled = self.resample(X_train,y_train,t,fold)
                y_train_resampled = y_resampled

                # 1) scale and/or reduce the data
                if self.verbose:
                    ptf('Scaling fold %d' % fold, self.logfile)
                X_scaled, scaler = self._scale_class(X_resampled,
                    self.detection_base_scaler,
                    self.detection_base_scaler_arguments)
                X_test_scaled = scaler.fit_transform(X_test)

                # 1A) reduce
                if self.verbose:
                    ptf('Reducing fold %d' % fold, self.logfile)
                X_reduced, reducer = self._reduce_class(X_scaled,
                    self.detection_base_reducer,
                    self.detection_base_reducer_arguments)
                X_test_reduced = reducer.transform(X_test_scaled)

                # 2) trigger_train
                if self.verbose:
                    ptf('Training fold %d' % fold, self.logfile)
                model, train_predictions, train_probabilities = \
                    self.trigger_train(X_reduced, y_train_resampled, fold, t)

                # 3) trigger_predict
                if self.verbose:
                    ptf('Predicting fold %d' % fold, self.logfile)
                test_predictions, test_probabilities = \
                    self.trigger_predict(model, X_test_reduced, fold, t)

                # 3A) store fold
                if self.verbose:
                    ptf('Storing fold %d' % fold, self.logfile)
                self._trigger_store_one_fold(
                    (train_predictions, train_probabilities),
                    (test_predictions, test_probabilities),
                    fold, t
                )

                # 3B) score one fold
                if self.verbose:
                    ptf('Scoring fold %d' % fold, self.logfile)
                # print t, y_resampled[0].shape,
                self._trigger_score_one_fold(y_train_resampled[0],
                    train_predictions, train_probabilities, t, testtrain='train', fold=fold)
                self._trigger_score_one_fold(y_test[0],
                    test_predictions, test_probabilities, t, testtrain='test', fold=fold)

                # stack probas
                if fold == 0:
                    y_test_probabilities = test_probabilities
                    y_train_probabilities = train_probabilities
                else:
                    y_test_probabilities = np.vstack((y_test_probabilities, test_probabilities))
                    y_train_probabilities = np.vstack((y_train_probabilities, train_probabilities))

                y_train_true_timestep.extend(y_train_resampled[0])
                y_train_predict_timestep.extend(train_predictions)
                y_test_true_timestep.extend(y_test[0])
                y_test_predict_timestep.extend(test_predictions)
            # 4) trigger_score
            if self.verbose:
                ptf('Scoring timestep %d' % t, self.logfile)
            self._trigger_score_one_fold(y_train_true_timestep,
                y_train_predict_timestep, y_train_probabilities, t, 'train')
            self._trigger_score_one_fold(y_test_true_timestep,
                y_test_predict_timestep, y_test_probabilities, t, 'test')


        # if self.trigger_only:
        #     return X_featurized


        # 4A) Write avg tau to file for each trial.  We will pass this into triggered series model

        ## UPDATES FOR V2 BEYOND THIS ##

        # 5) tranform tau

        # for t in range(0, self.max_postdetection_time, 20):
        return X_featurized
