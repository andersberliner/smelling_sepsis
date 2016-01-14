    sm = SeriesModel(
        # features_pickle = 'features_%s.pkl' % START_DT_STR,
        # fold_features_pickle = 'fold_features_%s.pkl' % START_DT_STR,
        # fold_features_test_pickle = 'fold_features_test_%s.pkl' % START_DT_STR,
        # featurizer_pickle = 'featurizer_%s.pkl' % START_DT_STR,
        # reducer_pickle = 'reducer_%s.pkl' % START_DT_STR,
        # scaler_pickle = 'scaler_%s.pkl' % START_DT_STR,
        on_disk = on_disk,
        load_state = load_state,
        load_time = load_time,
        runid = RUNID,
        featurizer_coldstart = reload_features,
        scaler_coldstart = reload_fold_features,
        reducer_coldstart = reload_fold_features,
        logfile = LOGFILE,
        use_last_timestep_results = use_last_timestep_results,
        color_scale = 'RGB',
        color_vector_type = 'DI',
        reference_time = 9,
        min_time = 3,
        detection_model = 'LRCV',
        detection_model_arguments = {'n_jobs':n_jobs},
        gram_model = 'LRCV',
        gram_model_arguments = {'n_jobs':n_jobs, 'multi_class':'ovr'},
        classification_model = 'LRCV',
        classification_model_arguments = {'n_jobs':n_jobs, 'multi_class':'ovr'},
        # detection_featurizer = 'sigmoid',
        # detection_featurizer_arguments = {},
        detection_featurizer = 'poly',
        detection_featurizer_arguments = {'n':4, 'n_jobs': n_jobs, 'gridsearch': False},
        # detection_featurizer = None,
        gram_featurizer = 'detection',
        classification_featurizer = 'detection',
        detection_reducer = 'pca',
        detection_reducer_arguments = {'n_components': 30},
        nfolds=nfolds,
        fold_size=fold_size
        )



# score and write to scores
self._score_one_timestep(y, y_predict_detection,
                         y_predict_gram, y_predict_classification,
                         number_of_times, fold)
# store results from this timestep
self._store_one_timestep((y_predict_detection, y_probabilities_detection),
                         (y_predict_gram, y_probabilities_gram),
                         (y_predict_classification, y_probabilities_classification),
                         number_of_times, fold)


    np_X_detection, np_X_gram, np_X_classification = X['detection'][number_of_times], X['gram'][number_of_times], X_classification['number_of_times']

def _train_one_timestep(self, t):
    y_train_pred = defaultdict(list)
    y_train_true = defaultdict(list)


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



#### FROM FIT - not used  ######

                # y_train_pred_detection = model_detection.predict(self.fold_features_test[fold]['detection'][t])
                # y_train_pred_gram = model_gram.predict(self.fold_features_test[fold]['gram'][t])
                # y_train_pred_classification = model_classification.predict(self.fold_features_test[fold]['classification'][t])


                # predict on Xtest by
                # scale fold data
                # reduce fold data
                # predict fold data
                X_test_detection = self.fold_features_test[fold]['detection'][t]
                X_test_gram = self.fold_features_test[fold]['gram'][t]
                X_test_classification = self.fold_features_test[fold]['classification'][t]

                # accumulate y_pred_test, y_true_test



                X_train = self.fold_features[fold]
                X_detection_train, X_detection_test, X_gram_train, X_gram_test, X_classification_train, X_classification_test
                self._fit_one_timestep(X, y, t, use_last_timestep_results)
                use_last_timestep_results = self.use_last_timestep_results




    def _fit_one_timestep(self, X, y, number_of_times, use_last_timestep_results=False):
        #
        np_X_detection, np_X_gram, np_X_classification = X['detection'][number_of_times], X['gram'][number_of_times], X_classification['number_of_times']
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
    # ==> DEPRECATED <== #
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




            ptf('\tnfolds: %s' % nfolds, LOGFILE)
            ptf('\tfold_size: %s' % fold_size, LOGFILE)
            ptf('\tuse_last_timestep_results: %s' % use_last_timestep_results, LOGFILE)
            ptf('\tmodel: %s' % model, LOGFILE)
            ptf('\tfeaturizer: %s' % featurizer, LOGFILE)
            ptf('\n', LOGFILE)
            ptf('\treload_data: %s' % reload_data, LOGFILE)
            ptf('\treload_features: %s' % reload_features, LOGFILE)
            ptf('\treload_fold_features: %s' % reload_fold_features, LOGFILE)
            ptf('\tn_jobs: %d\tn_cpus: %d' % (n_jobs, n_cpus), LOGFILE)
            ptf('\tdebug: %s' % debug, LOGFILE)
            ptf('\tprofile: %s' % profile, LOGFILE)
            ptf('\tverbose: %s' % verbose, LOGFILE)
            ptf('\n', LOGFILE)
            ptf('\tfeatures_pickle' % sm.features_pickle, LOGFILE)
            ptf('\tfold_features_pickle' % sm.fold_features_pickle, LOGFILE)

            model_file_name = 'sm_model_%s.pkl' % START_DT_STR
