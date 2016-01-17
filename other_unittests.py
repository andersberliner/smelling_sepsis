# other unittests.py


    if False:

        # i) SETUP
        sm.setup(X,y)
        if False:
            print 'Confusion_labels'
            print sm.confusion_labels
            print 'Predictions'
            print sm.fold_predictions
            print 'Predcitions test'
            print sm.fold_predictions_test
            print 'Probas'
            print sm.fold_probabilities
            print 'Probas test'
            print sm.fold_probabilities_test
            print 'Results'
            print sm.results
            print 'Scores'
            print sm.scores
            print 'Scores test'
            print sm.scores_test
            print 'Models'
            print sm.models
            print 'Featurizers'
            print sm.featurizers
            print 'Reducers'
            print sm.reducers
            print 'Scalers'
            print sm.scalers

        # 0) PREPROCESS #
        Z = sm.preprocess(X)
        if debug:
            print len(X), len(Z)
            print X.iloc[0][8:12,0:4], Z.iloc[0][8:12, 0:4]

        if False:
            # make r-objects to figure out calls of lme4, glmer in rstudio
            Xsub = sm._subset_data(Z, 50)

            # print stuff
            longdf = make_long_dataframe(Xsub, y, used_column_headers, Ntimes=50)

            # export_to_r_and_pickle(Xsub, y, used_column_headers, 50)


        sm.times = [15, 30, 50]
        sm.verbose = True
        sm.debug = True

        # 1) FEATURIZE #
        if reload_features:
            Zf = sm.featurize(Z, sm.featurizer_pickle)
            start = time.time()
            myfile = open('features.pkl', 'wb')
            pickle.dump(Zf, myfile)
            myfile.close()
            end = time.time()
            ptf('Pickled in %d s' % (start-end), LOGFILE)
        else:
            start = time.time()
            myfile = open('features.pkl', 'rb')
            Zf = pickle.dump(myfile)
            myfile.close()
            sm.features = Zf
            end = time.time()
            ptf('Loaded features in %d s' % (start-end), LOGFILE)
        if debug:
            print 'Featurizers'
            print sm.featurizers

        # 2) SCALE #
        Zs = sm.scale(Zf, sm.scaler_pickle)
        if debug:
            print 'Scalers'
            print sm.scalers

        # 3) REDUCE #
        Zr = sm.reduce(Zs, sm.reducer_pickle)
        if debug:
            print 'Reducers'
            print sm.reducers
