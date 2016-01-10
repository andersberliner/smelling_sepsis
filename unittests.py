# unittests.py
# Anders Berliner
# 20160110

from seriesmodel import SeriesModel
from featurizer import PolynomialFeaturizer
from timeseriesplotter import SpotTimePlot

def run_unittests(X_test, y_test, verbose=False):
    print 'Do some unit tests on seriesmodel, featurizer...'
    print '0) Set-up (prepare data)'
    sm = SeriesModel(reference_time=9)

    if False:
        print 'A) seriesmodel preprocessing'
        if verbose:
            print X_test.iloc[0][0:5, 0:4]
        print 'DI, reftime = 2'
        DI = sm.preprocess(X_test)
        if verbose:
            print DI.iloc[0][0:5, 0:4]
        print 'DII, reftime = 3'
        sm = SeriesModel(color_vector_type='DII', reference_time=3)
        DII = sm.preprocess(X_test)
        if verbose:
            print DII.iloc[0][0:5, 0:4]

    sm.confusion_labels = sm._build_confusion_labels(y_test)
    Z = sm._prepare_data(X_test)

    if False:
        print 'B) Set-up results dataframes'
        if verbose:
            print 'Predictions'
            print sm.predictions.head()
            print 'Probabilities'
            print sm.probabilities.head()
            print 'Metrics'
            print sm.metrics
            print 'Confusion Labels'
            print sm.confusion_labels
            print 'Scores'
            print sm.scores

    print '1) Fit one timestep...'
    sm._fit_one_timestep(Z, y_test, 10)

    print 'Fit another'
    sm._fit_one_timestep(Z, y_test, 20)

    print 'And another'
    sm._fit_one_timestep(Z, y_test, 30)

    print 'Predict these timesteps'
    sm.times = [10,20,30]
    yd,yg,yc = sm.predict(X_test)

    print 'Score timesteps'
    results = sm.score(y_test, verbose=True)

    print 'Try again using the full workthrough'

    sm = SeriesModel(reference_time=9, min_time=3)
    sm.fit(X_test, y_test, verbose=True, debug=True)

    yd, yg, yc = sm.predict(X_test.iloc[0:10])
    results = sm.score(y_test.iloc[0:10])


    if verbose:
        print 'Check shape of results'
        for c in sm.results.columns:
            print c, sm.results[c].iloc[0].shape

        print 'Check shape of predictions'
        for k, v in sm.predictions.iteritems():
            print k, len(v)

        print 'Check shape of probabilities'
        for k, v in sm.probabilities.iteritems():
            print k, v.shape

        print 'Check confusion matrices for sense'
        print sm.confusion_labels['gram']
        print sm.scores['gram']['micro'].iloc[0]['confusion_matrix']
        print sm.scores['gram']['n'].iloc[0]['confusion_matrix']


    if False:
        print 'A) Subset data'
        nt = 4
        X_sub = sm._subset_data(Z, nt)

        if verbose:
            print X_sub.shape, X_sub.iloc[0].shape, Z.shape, Z.iloc[0].shape
            print Z.iloc[0][0:5,0:5]
            print X_sub.iloc[0][:,0:5]

        print 'B) Featurize data'
        Xf, featurizer = sm._featurize_class(X_sub, 'poly', {'n':3})


        print 'i) Try all the same featurizer'
        Xd, Xg, Xc = sm._featurize(X_sub, nt)

        if verbose:
            print Xd.head()
            print Xg.head()
            print Xc.head()

        print 'ii) Try all different featurizers'
        sm.gram_base_featurizer_arguments = {'n':6}
        sm.gram_base_featurizer = 'poly'
        sm.classification_base_featurizer = None
        Xd2, Xg2, Xc2 = sm._featurize(X_sub, nt)

        if verbose:
            print Xd2.head()
            print Xg2.head()
            print Xc2.head()

        print 'C) Run detection model'
        nt = 30
        X_sub = sm._subset_data(Z, nt)

        Xd, Xg, Xc = sm._featurize(X_sub, nt)

        print 'i) Convert featurized data to np array'
        np_Xd = sm._pandas_to_numpy(Xd)
        if verbose:
            print np_Xd.shape
            print y_test['detection'].values.shape

        print 'ii) Train model'
        detection_model = sm._fit_class(np_Xd, y_test['detection'].values,
                                'LR',
                                sm.detection_base_model_arguments,
                                step=('detection t=%d' % nt))

        if verbose:
            print 'Inspect stored models, featurizers'
            print sm.models
            print sm.featurizers



    if False:
        print '\n\n2) PolynomialFeaturizer'
        start = time.time()

        PF = PolynomialFeaturizer(n=4, reference_time=2, verbose=True)
        mycoefs, myscores = PF.fit_transform(DI)
        # myscores = PF.score()
        DI_pred = PF.predict(DI, mycoefs)

        end = time.time()
        print 'Featurized, predicted %d test trails in %d seconds' % (len(y_test),(end-start))

        # need to add curve visualization here to do a sanity check
        # on the data
        print 'Curve visualization...'
        STP = SpotTimePlot(y_test, used_column_headers)
        STP.plot_fits(DI, DI_pred)
    return sm
