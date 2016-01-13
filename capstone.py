# capstone.py
# Anders Berliner
# 20160104

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import ntpath
import os
import re
import datetime
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
import time
# import rpy2
from seriesmodel import SeriesModel
from featurizer import PolynomialFeaturizer
from timeseriesplotter import SpotTimePlot
from collections import defaultdict
from unittests import run_unittests
import multiprocessing
from output_capstone import print_to_file_and_terminal as ptf
from utils_capstone import load_data, split_train_test, export_to_r_and_pickle, \
        my_pickle, my_unpickle
from capstone_r_tools import make_long_dataframe
from sklearn.linear_model import LogisticRegression
import pickle
from itertools import izip

# use pympler to do memory diagnosis
from pympler import asizeof, tracker, classtracker

START_DT = datetime.datetime.now()
START_DT_STR = START_DT.strftime('%Y-%m-%d_%H-%M-%S')
LOGFILE = open('log_%s.txt' % START_DT_STR, 'w')
PICKLE_NAMES = ['Xdf.pkl', 'ydf.pkl', 'used_column_headers.pkl']

if __name__ == '__main__':
    # Capstone run conditions
    reload_data = False
    pickle_data = False # save loaded data to a pickle for greater loading efficiency
    unittests = False
    profile = False # do memory profiling

    # Model conditions
    verbose = True # how much output
    debug = True # subset timeseries
    n_cpus = multiprocessing.cpu_count()
    n_jobs = n_cpus # to see if more ram is used for more cpus
    use_last_timestep_results = False # feed forward probabilities
    model = 'LR' # base model used for LR
    featurizer = 'poly' # base model used for featurization
    nfolds = 10 # number of cross_validation folds
    fold_size = 0.1 # size of cross_validation folds

    ptf('====> Starting job ID: %s <====' % START_DT_STR, LOGFILE)
    ptf('\tn_jobs: %d\tn_cpus: %d' % (n_jobs, n_cpus), LOGFILE)
    ptf('\tdebug: %s' % debug, LOGFILE)
    ptf('\tprofile: %s' % profile, LOGFILE)
    ptf('\tverbose: %s' % verbose, LOGFILE)
    ptf('\tuse_last_timestep_results: %s' % use_last_timestep_results, LOGFILE)
    ptf('\tmodel: %s' % model, LOGFILE)
    ptf('\tfeaturizer: %s' % featurizer, LOGFILE)
    ptf('\tnfolds: %s' % nfolds, LOGFILE)
    ptf('\tfold_size: %s' % fold_size, LOGFILE)

    if profile:
        # set-up some tracking statements from pympler
        tr = tracker.SummaryTracker()
        tr_sm = classtracker.ClassTracker()
        # tr_pf = classtracker.ClassTracker()
        # tr_stp = classtracker.ClassTracker()
        # tr_lr = classtracker.ClassTracker()

        tr_sm.track_class(SeriesModel)
        # tr_pf.track_class(PolynomialFeaturizer)
        # tr_stp.track_class(SpotTimePlot)
        # tr_lr.track_class(LogisticRegression)

        tr_sm.create_snapshot()
        # tr_pf.create_snapshot()
        # tr_stp.create_snapshot()
        # tr_lr.create_snapshot()

    # data loading
    if reload_data:
        root_folder = 'Shared Sepsis Data'
        csv_filename = os.path.join(root_folder, 'Sepsis_JCM.csv')

        X, y, used_column_headers, df, df_raw = load_data(root_folder, csv_filename, verbose=False, LOGFILE=LOGFILE)
        # ptf( '\n>> Test/train split...', LOGFILE)
        # X_train, X_test, y_train, y_test = split_train_test(X,y, test_size=0.20, LOGFILE=LOGFILE)

        # NOTE: can get back all of the data via df.ix[] whatever the index is
        # in y_train, y_test, etc.

        ## PICKLE ##
        if pickle_data:
            start = time.time()
            ptf( '\n>> Pickling data ...\n', LOGFILE)
            for z, zname in izip([X,y,used_column_headers], PICKLE_NAMES):
                my_pickle(z, zname)
            end = time.time()
            ptf( 'Data pickled in %d seconds (%d total trials)' % ((end-start), len(X)), LOGFILE)

    else:
        start = time.time()

        ptf( '\n>> Unpickling data ...\n', LOGFILE)
        X = my_unpickle(PICKLE_NAMES[0])
        y = my_unpickle(PICKLE_NAMES[1])
        used_column_headers = my_unpickle(PICKLE_NAMES[2])

        end = time.time()
        ptf( 'Data unpickled in %d seconds (%d total trials)' % ((end-start), len(X)), LOGFILE)




    sm = SeriesModel(
        featurizer_coldstart = False,
        logfile = None,
        use_last_timestep_results = use_last_timestep_results,
        color_scale = 'RGB',
        color_vector_type = 'DI',
        reference_time = 9,
        min_time = 3,
        detection_model = 'LR',
        detection_model_arguments = {'n_jobs':n_jobs},
        gram_model = 'LR',
        gram_model_arguments = {'n_jobs':n_jobs, 'multi_class':'ovr'},
        classification_model = 'LR',
        classification_model_arguments = {'n_jobs':n_jobs, 'multi_class':'ovr'},
        detection_featurizer = 'poly',
        detection_featurizer_arguments = {'n':4},
        gram_featurizer = 'detection',
        classification_featurizer = 'detection',
        nfolds=nfolds,
        fold_size=fold_size
        )

    sm.setup(X,y)
    if debug:
        print sm.confusion_labels
        print sm.predictions
        print sm.results



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
    Zf = sm.featurize(Z, sm.featurizer_pickle)
    Zs = sm.scale(Zf, sm.scaler_pickle)
    Zr = sm.reduce(Zs, sm.reducer_pickle)





    if False:
        start = time.time()
        sm.fit(X_train,y_train, verbose=verbose, debug=debug)

        end = time.time()

        ptf( '\n\n>> Model fit (%d times, %d samples) in %d seconds (%d mins) <<\n\n' % (len(sm.times), len(y_train), (end-start), (end-start)/60.0), LOGFILE)

        # Pickle model
        train_file_name = 'sm_train_%s.pkl' % START_DT_STR
        train_file = open(train_file_name, 'wb')
        ptf('\n>> Writing train results to %s' % train_file_name, LOGFILE)
        pickle.dump(sm, train_file, -1)
        train_file.close()

        # predict
        start = time.time()
        yd, yg, yc = sm.predict(X_test, verbose=verbose)
        end = time.time()
        ptf( '\n\n>> Model predictions (%d times, %d samples) in %d seconds (%d mins) <<\n\n' % (len(sm.times), len(y_test), (end-start), (end-start)/60.0), LOGFILE)

        start = time.time()
        results = sm.score(y_test, verbose=verbose)
        end = time.time()
        ptf( '\n\n>> Model scores (%d times, %d samples) in %d seconds (%d mins) <<\n\n' % (len(sm.times), len(y_test), (end-start), (end-start)/60.0), LOGFILE)

        # Pickle model results
        test_file_name = 'sm_test_%s.pkl' % START_DT_STR
        test_file = open(test_file_name, 'wb')
        ptf('\n>> Writing test results to %s' % test_file_name, LOGFILE)
        pickle.dump(sm, test_file, -1)
        test_file.close()


        if profile:
            ptf( '\nSERIESMODEL profiling', LOGFILE)
            ptf( 'Look at size of seriesmodel object', LOGFILE)
            ptf( asizeof.asizeof(sm), LOGFILE)
            ptf( asizeof.asized(sm, detail=1).format(), LOGFILE)

            ptf( 'Look at how the SeriesModel class is doing', LOGFILE)
            tr_sm.create_snapshot()
            tr_sm.stats.print_summary()
            tr_sm.stats.print_summary() >> LOGFILE

            # print '\nPOLYNOMIALFEATURIZER profiling'
            # print 'Size of PF object'
            # print asizeof.asizeof(sm.featurizers['gram'][50])
            # print asizeof.asized(sm.featurizers['gram'][50], detail=1).format()
            # print 'Look at how the PolynomialFeaturizer class is doing'
            # tr_pf.create_snapshot()
            # tr_pf.stats.print_summary()
            #
            # print '\nSPOTTIMEPLOT profiling'
            # print 'Size of a STP object'
            # DI = sm.preprocess(X_test)
            # PF = PolynomialFeaturizer(n=4, reference_time=2, verbose=True)
            # mycoefs, myscores = PF.fit_transform(DI)
            # # myscores = PF.score()
            # DI_pred = PF.predict(DI, mycoefs)
            # STP = SpotTimePlot(y_test, used_column_headers)
            # STP.plot_fits(DI, DI_pred)
            # print asizeof.asizeof(STP)
            # print asizeof.asized(STP, detail=1).format()
            #
            # print 'Look at how stp is doing'
            # tr_stp.create_snapshot()
            # tr_stp.stats.print_summary()
            #
            # print '\nLR profiling'
            # print 'Sizer of an LR object'
            # print asizeof.asizeof(sm.models['classification'][50])
            # print asizeof.asized(sm.models['classification'][50], detail=1).format()
            #
            # print 'Look at how LR is doing'
            # tr_lr.create_snapshot()
            # tr_lr.stats.print_summary()

            ptf( 'PROFILING', LOGFILE)
            ptf( 'Look at memory leaks up to this point', LOGFILE)
            tr.print_diff() >> LOGFILE
            tr.print_diff()

    ### Unittests ###
    if False:
        sm_unit = run_unittests(X_test, y_test, verbose=False)


    LOGFILE.close()
