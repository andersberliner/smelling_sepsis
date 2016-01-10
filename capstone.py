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
import rpy2
from seriesmodel import SeriesModel
from featurizer import PolynomialFeaturizer
from timeseriesplotter import SpotTimePlot
from collections import defaultdict
from unittests import run_unittests

def timestamp_interpretter(x):
    # TODO - fix regex for timestamps of the type:
    #   2012-Apr-17_23-42-03
    # all image names are of the type:
    #   FrameName_YYYY-MM-DD_HH-MM-SS
    time_str_regex = r'.*_([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})'

    time_str = re.findall(time_str_regex, x)[0]
    time_stamp = datetime.datetime.strptime(time_str, '%Y-%m-%d_%H-%M-%S')
    # CONVERT TO MINUTES (or HOURS?) ELAPSED
    return time_stamp

def load_spots_files(x, root_folder, column_headers,
        columns_to_drop, fname = 'spots.txt', verbose=False):
    # csv file created in a windows enivornment
    relpath = x['Folder'].replace(ntpath.sep, os.sep)
    data_file = os.path.join(root_folder, relpath, fname)
    if verbose:
        print data_file
    # load data file
    mini_df = pd.read_table(data_file, header=None)
    # some files have extra tab, creating extra data column.
    # Strip them
    if mini_df.values.shape[1]>241:
        mini_df.drop(241, inplace=True, axis=1)

    # add column_headers to df
    mini_df.columns = column_headers

    # drop spots
    mini_df.drop(columns_to_drop, inplace=True, axis=1)


    # convert filenames to minutes
    # mini_df['time'] = mini_df['time'].apply(timestamp_interpretter)
    ntimes = len(mini_df)
    mini_df['time'] = np.arange(0, 20*ntimes, 20)

    x['data'] = mini_df
    return x

def build_labels(x):
    pass

def create_column_headers(nspots=80, colors=['R', 'G', 'B']):
    column_headers = ['time']
    for x in xrange(1, nspots+1):
        for color in colors:
            column_headers.append(str(x) + color)
    return column_headers

def populate_columns_to_drop(other_spots = [], colors=['R', 'G', 'B']):
    # strip blanks, faducials
    columns_to_drop = []
    for spot in [1, 10, 12, 19, 62, 71, 80]:
        for color in colors:
            columns_to_drop.append(str(spot) + color)
    # strip other spots that are shown to be problematic
    for spot in other_spots:
        for color in colors:
            columns_to_drop.append(str(spot) + color)

    return columns_to_drop

def create_labels_dictionaries(species_info_filename = 'species_to_gram2.csv'):
    label_df = pd.read_csv(species_info_filename, index_col=0)

    # label_dictionaries = {{column_name: label_df[column_name].to_dict()} for column_name in label_df.columns}
    label_dictionaries = {}
    for column_name in label_df.columns:
        label_dictionaries[column_name] = label_df[column_name].to_dict()

    return label_dictionaries

def create_labels(df, label_dictionaries):
    df['label'] = df['Genus'] + '_' + df['Species']
    df['label'] = df['label'].apply(lambda x: 'Control' if x == 'control_control' else x)

    for k, v in label_dictionaries.iteritems():
        if k in ['url', 'pic']: # use these for display purposes
            pass
        df[k] = df['label'].apply(lambda x: v[x])

    return df

def split_train_test(X, y, test_size=0.25, verbose=False):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,
    #                                                 stratify=y['classification'].unique())
    sss = StratifiedShuffleSplit(y=y['classification'], n_iter=1, test_size=test_size, random_state=1)
    for train_index, test_index in sss:
        print 'Train: ', len(train_index)
        print 'Test: ', len(test_index)

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    if verbose:
        print '\nTEST summary:'
        print y_test.groupby('classification').count()
        print '\nTRAIN summary:'
        print y_train.groupby('classification').count()

    return X_train, X_test, y_train, y_test

def prepare_data_frame(df_raw):
    df = df_raw.copy()
    used_column_headers = df['data'].iloc[0].columns

    df['data'] = df['data'].apply(lambda x: x.values)
    return df, used_column_headers

def find_nans(df):
    return df[df.apply(lambda x: int(np.sum(np.isnan(x)))) > 0]


def find_bad_rows(x, Imax, Imin):
    x = x.drop('time', axis=1)
    bad_rows = []
    for i, row in x.iterrows():
        # if i%100 == 0:
        #     print i

        if (np.sum(Imin >= row) + np.sum(row >= Imax)) > 0:
            bad_rows.append(i)

    return bad_rows

def find_data_anomalies(df, Imax=4096.0, Imin=0.0):
    anomalies = defaultdict(list)

    an_df = df.apply(lambda x: (find_bad_rows(x['data'], Imax, Imin), x['Folder']), axis=1)
    #print an_df
    an_df = an_df.map(lambda x: x if x[0] else False)
    #print an_df
    an_df = an_df[an_df > 1]

    return an_df

def load_data(root_foler, csv_filename, verbose=False):
    print 'Loading csv...'
    df_raw = pd.read_csv(csv_filename)

    # only work with "good" trials
    df_raw = df_raw[df_raw['Ignore'] != True]

    column_headers = create_column_headers()
    columns_to_drop = populate_columns_to_drop()

    start = time.time()
    print 'Loading data files...'
    df_raw = df_raw.apply(lambda x: load_spots_files(x, root_folder, column_headers,
                                                columns_to_drop), axis=1)
    end = time.time()
    print 'Data loaded in %d seconds (%d total trials)' % ((end-start), len(df_raw))

    # DATA INSPECTION - finding values outside of 0, 4096
    # Reference_time must be greater than 1 if we use DII
    # see code and snippet below
    # All trials from the same day and at the same time (1: 20 minutes)
    # ==> instrumentation error at that time
    if verbose:
        start = time.time()
        print 'Finding anomalous trials...'
        an_df = find_data_anomalies(df_raw)
        end = time.time()
        print 'Anomalous trials found in %d seconds (%d trials):' % ((end-start), len(an_df))
        print an_df

        '''
        Finding anomalous trials...
        Anomalous trial found in 44 seconds (13 trials):
        372          ([1], 20120504\BCB\E. coli 25922 10 CFU\F1)
        373         ([1], 20120504\BCB\E. coli 25922 10 CFU\F16)
        374        ([1], 20120504\BCB\S. aureus 29213 10 CFU\F7)
        375       ([1], 20120504\BCB\S. aureus 29213 10 CFU\F12)
        376       ([1], 20120504\BCB\S. aureus 29213 10 CFU\F21)
        377     ([1], 20120504\BCB\S. maltophilia Clinical A\F4)
        378    ([1], 20120504\BCB\S. maltophilia Clinical A\F13)
        379    ([1], 20120504\BCB\S. maltophilia Clinical A\F18)
        380    ([1], 20120504\BCB\S. maltophilia Clinical A\F23)
        381     ([1], 20120504\BCB\S. maltophilia Clinical B\F5)
        382    ([1], 20120504\BCB\S. maltophilia Clinical B\F10)
        383    ([1], 20120504\BCB\S. maltophilia Clinical B\F15)
        384    ([1], 20120504\BCB\S. maltophilia Clinical B\F20)
        '''

    # re-order 'data' part of frame for convenience
    # currently exists as a data frame with name columns
    # time, 2R, 2G, 2B .... 79R, 79G, 79B
    # need to be able to manipulate data as numpy arrays
    # keep column headers around for later use
    df, used_column_headers = prepare_data_frame(df_raw)

    print 'Creating labels...'
    label_dictionaries = create_labels_dictionaries()
    df = create_labels(df, label_dictionaries)

    # drop unwanted labels
    df = df[df['Ignore_label'] != True]

    X = df['data']
    y = df[['classification', 'gram', 'detection']]

    if verbose:
        print '\nSummary counts after cleaning:'
        print y.groupby('gram').count()
        print y.groupby('detection').count()
        print y.groupby('classification').count()

    return X, y, used_column_headers, df, df_raw

if __name__ == '__main__':
    verbose = True
    quickload = True

    # data loading
    if True:
        root_folder = 'Shared Sepsis Data'
        csv_filename = os.path.join(root_folder, 'Sepsis_JCM.csv')

        X, y, used_column_headers, df, df_raw = load_data(root_folder, csv_filename, verbose=False)
        print 'Test/train split...'
        X_train, X_test, y_train, y_test = split_train_test(X,y, test_size=0.20)


        # NOTE: can get back all of the data via df.ix[] whatever the index is
        # in y_train, y_test, etc.

    if True:
        start = time.time()
        sm = SeriesModel(
            color_scale = 'RGB',
            color_vector_type = 'DI',
            reference_time = 9,
            min_time = 3,
            detection_model = 'LR',
            detection_model_arguments = {'n_jobs':-1},
            gram_model = 'LR',
            gram_model_arguments = {'n_jobs':-1, 'multi_class':'ovr'},
            classification_model = 'LR',
            classification_model_arguments = {'n_jobs':-1, 'multi_class':'ovr'},
            detection_featurizer = 'poly',
            detection_featurizer_arguments = {'n':4},
            gram_featurizer = 'detection',
            classification_featurizer = 'detection'
            )

        sm.fit(X_train,y_train, verbose=verbose, debug=True)

        end = time.time()

        print '\n\n>>Model fit (%d times, %d samples) in %d seconds (%d mins)<<' % (len(sm.times), len(y_train), (end-start), (end-start)/60.0)

        # predict
        start = time.time()
        yd, yg, yc = sm.predict(X_test, verbose=True)
        end = time.time()
        print '\n\n>>Model predictions (%d times, %d samples) in %d seconds (%d mins)<<' % (len(sm.times), len(y_test), (end-start), (end-start)/60.0)

        start = time.time()
        results = sm.score(y_test, verbose=True)
        end = time.time()
        print '\n\n>>Model scores (%d times, %d samples) in %d seconds (%d mins)<<' % (len(sm.times), len(y_test), (end-start), (end-start)/60.0)



    ### Unittests ###
    if False:
        sm_unit = run_unittests(X_test, y_test, verbose=False)
