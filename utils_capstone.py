# utils_capstone.py
# Anders Berliner
# 20160111

import datetime
import ntpath
import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from output_capstone import print_to_file_and_terminal as ptf
import time

# from rpy2.robjects import r, pandas2ri, numpy2ri
import pickle

def my_to_r(x, xname):
    ro = numpy2ri.numpy2ri(x)
    r.assign(xname, ro)
    r_str = "save(%s, file='%s.gzip', compress=TRUE)" % (xname, xname)
    r(r_str)

def my_pickle(x, xname):
    myfile = open(xname, 'wb')
    pickle.dump(x, myfile, -1)
    myfile.close()

def my_unpickle(xname):
    myfile = open(xname, 'rb')
    x = pickle.load(myfile)
    myfile.close()
    return x

def export_to_r_and_pickle(X, y, used_column_headers, t):

    pandas2ri.activate()

    my_pickle(X, 'X.pkl')
    my_to_r(X, 'Xdf')
    # rX = pandas2ri(X)
    # r.assign('X', rX)
    # r("save(X, file='Xdf.gzip', compress=TRUE)")

    # y convert y to 0 or 1 labels for each timestep
    Y = y['detection'].apply(lambda x: np.ones((t,1)) if x else np.zeros((t,1)))
    my_pickle(Y, 'Y.pkl')
    my_to_r(Y, 'Ydf')
    # rY = pandas2ri(Y)
    # r.assign('Y', rY)
    # r("save(Y, file='Ydf.gzip', compress=TRUE)")

    # extract the spots for use
    spots = [int(x[:-1]) for x in used_column_headers[1:]]
    spots = (np.array(spots)).reshape(-1,1)

    my_pickle(spots, 'spots.pkl')
    my_to_r(spots, 'spots')

    # rspots = numpy2ri(spots)
    # r.assign('spots', rspots)
    # r("save(spots, file='spots.gzip', compress=TRUE)")

    # make time vector
    time = (20*np.arange(0,t,1)).reshape(-1,1)
    my_pickle(time, 'time.pkl')
    my_to_r(time, 'time')
    # rtime = numpy2ri(time)
    # r.assign('time', rtime)
    # r("save(time, file='time.gzip', compress=TRUE)")

    # make id vector
    trial_id = (np.array(y.index.values)).reshape(-1,1)

    my_pickle(trial_id, 'trial_id.pkl')
    my_to_r(trial_id, 'id')
    # rid = numpy2ri(trial_id)
    # r.assign('id', rid)
    # r("save(id, file='id.gzip', compress=TRUE)")

    numpy2ri.deactivate()
    pandas2ri.deactivate()

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
        columns_to_drop, fname = 'spots.txt', verbose=False, LOGFILE=None):
    # csv file created in a windows enivornment
    relpath = x['Folder'].replace(ntpath.sep, os.sep)
    data_file = os.path.join(root_folder, relpath, fname)
    if verbose:
        ptf(['Loading data from', data_file], LOGFILE)
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

def split_train_test(X, y, test_size=0.25, verbose=False, LOGFILE=None):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,
    #                                                 stratify=y['classification'].unique())
    sss = StratifiedShuffleSplit(y=y['classification'], n_iter=1, test_size=test_size, random_state=1)
    for train_index, test_index in sss:

        ptf(['Train: ', len(train_index)], LOGFILE)
        ptf(['Test: ', len(test_index)], LOGFILE)

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    if verbose:
        ptf( '\nTEST summary:', LOGFILE)
        ptf( y_test.groupby('classification').count(), LOGFILE)
        ptf( '\nTRAIN summary:', LOGFILE)
        ptf( y_train.groupby('classification').count(), LOGFILE)

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

def load_data(root_folder, csv_filename, verbose=False, LOGFILE=None):
    ptf( '\n>> Loading csv...\n', LOGFILE)
    df_raw = pd.read_csv(csv_filename)

    # only work with "good" trials
    df_raw = df_raw[df_raw['Ignore'] != True]

    column_headers = create_column_headers()
    columns_to_drop = populate_columns_to_drop()

    start = time.time()
    ptf( '\n>> Loading data files...\n', LOGFILE)
    df_raw = df_raw.apply(lambda x: load_spots_files(x, root_folder, column_headers,
                                                columns_to_drop, LOGFILE=LOGFILE), axis=1)
    end = time.time()
    ptf( 'Data loaded in %d seconds (%d total trials)' % ((end-start), len(df_raw)), LOGFILE)

    # DATA INSPECTION - finding values outside of 0, 4096
    # Reference_time must be greater than 1 if we use DII
    # see code and snippet below
    # All trials from the same day and at the same time (1: 20 minutes)
    # ==> instrumentation error at that time
    if verbose:
        start = time.time()
        ptf( 'Finding anomalous trials...', LOGFILE)
        an_df = find_data_anomalies(df_raw)
        end = time.time()
        ptf( 'Anomalous trials found in %d seconds (%d trials):' % ((end-start), len(an_df)), LOGFILE)
        ptf( an_df, LOGFILE)

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

    ptf( 'Creating labels...', LOGFILE)
    label_dictionaries = create_labels_dictionaries()
    df = create_labels(df, label_dictionaries)

    # drop unwanted labels
    df = df[df['Ignore_label'] != True]

    X = df['data']
    y = df[['classification', 'gram', 'detection']]

    if verbose:
        ptf( '\nSummary counts after cleaning:', LOGFILE)
        ptf( y.groupby('gram').count(), LOGFILE)
        ptf( y.groupby('detection').count(), LOGFILE)
        ptf( y.groupby('classification').count(), LOGFILE)

    return X, y, used_column_headers, df, df_raw
