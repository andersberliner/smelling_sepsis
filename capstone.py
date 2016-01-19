'''
capstone.py
Anders Berliner
20160104

This is the "job-runner" containing the work-flow to create and test a
series model.

Most model parameters should be set in your json (e.g. run001.json).
Go to ### RUN CONDITIONS ### for things you might tweak for restarting a job, etc.
'''

import os
import re
import datetime
import time
from seriesmodel import SeriesModel
from tsm import TriggeredSeriesModel
from unittests import run_unittests
from tsm_unittests import run_tsm_unittests
from visualization_capstone import make_series_plots, make_trigger_plots
import multiprocessing
from output_capstone import print_to_file_and_terminal as ptf
from utils_capstone import load_data, split_train_test, export_to_r_and_pickle, \
        my_pickle, my_unpickle
from capstone_r_tools import make_long_dataframe
import pickle
from itertools import izip
import json
import shutil
from sys import argv
# use pympler to do memory diagnosis
from pympler import asizeof, tracker, classtracker

START_DT = datetime.datetime.now()
START_DT_STR = START_DT.strftime('%Y-%m-%d_%H-%M-%S')
PICKLE_NAMES = ['Xdf.pkl', 'ydf.pkl', 'used_column_headers.pkl']
RUNID = 'run200' # default name if now specified on command line
RUNTYPE = 'trigger'
### RUN CONDITIONS ###
MODELFILENAME = 'sm'
# LOGFILENAME = None
PICKLE_DATA = True # save loaded data to a pickle for future loading efficiency
DO_TESTS = False # run unittests instead of the main model
PROFILE = False # do memory profiling
verbose = True # how much output
debug = True # subset timeseries to just 3 points (T)
RELOAD = False # load raw data from folders (T) or use pickled data (F)
n_cpus = multiprocessing.cpu_count()

####################

def main(RUNID='run001', START_DT_STR=None, MODELFILENAME='sm', PICKLE_DATA=False,
    DO_TESTS=False, PROFILE=False, verbose=False, debug=False,
    RELOAD=False, n_cpus=1,
    PICKLE_NAMES=['Xdf.pkl', 'ydf.pkl', 'used_column_headers.pkl']):
    # paste stuff in later
    pass


def ascii_encode_dict(d):
    '''
    Re-encodes a json dictionary with (possibly) unicode k or v as an ascii
    encdoed python dictionary

    IN:  d - a dictionary loaded from a json file
    OUT: ascii-encoded python dictionary
    '''
    ascii_encode = lambda x: x.encode('ascii') if isinstance(x, unicode) else x
    # return dict(map(ascii_encode, pair) for pair in data.items())
    return {ascii_encode(k): ascii_encode(v) for k, v in d.iteritems()}

def command_line_process(RUNID):
    # do command-line processing to find RUNID
    if len(argv) > 2:
        print 'ERROR - too many command-line arguments'
        print 'Usage capstone RUNID'
        print 'Where RUNID is the folder and file name, minus extension of'
        print 'the json containing the run conditions'
    elif len(argv) == 2:
        RUNID = argv[1]
    else:
        print 'NOTE: you can specify folder, json name on command line'
        print 'Using default: %s' % RUNID

    return RUNID

def create_logfile(RUNID, LOGFILENAME):
    # creates a directory to store this runs work and copies the run params there
    if not os.path.exists('./' + RUNID):
        os.makedirs('./' + RUNID)
    if LOGFILENAME:
        LOGFILE = open('./' + RUNID + '/' + LOGFILENAME, 'w')
    else:
        LOGFILE = NONE

    # copy run_params to that folder
    shutil.copyfile((RUNID + '.json'), ('./' + RUNID + '/' + RUNID + '.json'))

    return LOGFILE

def start_memory_profiling():
    '''set-up some tracking statements from pympler'''
    tr = tracker.SummaryTracker()
    tr_sm = classtracker.ClassTracker()
    tr_sm.track_class(SeriesModel)
    tr_sm.create_snapshot()

    return tr, tr_sm

def print_memory_profiles(sm, tr, tr_sm, LOGFILE = None):
    '''prints report on memory profiles'''
    ptf( '\nSERIESMODEL profiling', LOGFILE)
    ptf( 'Look at size of seriesmodel object', LOGFILE)
    ptf( asizeof.asizeof(sm), LOGFILE)
    ptf( asizeof.asized(sm, detail=1).format(), LOGFILE)

    ptf( 'Look at how the SeriesModel class is doing', LOGFILE)
    tr_sm.create_snapshot()
    tr_sm.stats.print_summary()
    tr_sm.stats.print_summary() >> LOGFILE

    ptf( 'PROFILING', LOGFILE)
    ptf( 'Look at memory leaks up to this point', LOGFILE)
    tr.print_diff() >> LOGFILE
    tr.print_diff()

def reload_data(LOGFILE = None, PICKLE_DATA = True,
    root_folder = 'Shared Sepsis Data', csv_filename = 'Sepsis_JCM.csv'):
    '''
    Reloads raw_data from folders.
    IN:
        LOGFILE -  fileobj - an open text file where logs are written
        PICKLE_DATA - bool - whether to pickle data once loaded
        root_folder - str - relative path to top level folder for all data and csv_file
        csv_filename - str - name of csv file containing the trial labels and locations
    '''
    csv_file = os.path.join(root_folder, csv_filename)

    X, y, used_column_headers, df, df_raw = load_data(root_folder, csv_file,
        verbose=False, LOGFILE=LOGFILE)

    # pickle data for later loading efficiency
    if PICKLE_DATA:
        start = time.time()
        ptf( '\n>> Pickling data ...\n', LOGFILE)
        for z, zname in izip([X,y,used_column_headers], PICKLE_NAMES):
            my_pickle(z, zname)
        end = time.time()
        ptf( 'Data pickled in %d seconds (%d total trials)' % ((end-start),
            len(X)), LOGFILE)

    return X, y, used_column_headers, df, df_raw

def print_job_info(run_params, n_jobs, n_cpus, RUNID, START_DT_STR, LOGFILE = None,
    debug = False, profile=False, verbose = True, start = True):
    # Outputs header, footer describing job conditions

    if start:
        ptf('====> Starting job ID: %s_%s <====' % (RUNID, START_DT_STR), LOGFILE)
    else:
        ptf('====> Completed job ID: %s_%s <====' % (RUNID, START_DT_STR), LOGFILE)
    ptf('\tn_jobs: %d\tn_cpus: %d' % (n_jobs, n_cpus), LOGFILE)
    ptf('\tdebug: %s' % debug, LOGFILE)
    ptf('\tprofile: %s' % profile, LOGFILE)
    ptf('\tverbose: %s' % verbose, LOGFILE)
    for k, v in run_params.iteritems():
        ptf('\t%s: %s' % (k,v), LOGFILE)

def print_run_details(X, sm, LOGFILE=None):
    # prints other run details
    ptf('\n\n>> Run details <<')
    ptf('\tntrials: %d' % len(X), LOGFILE)
    ptf('\tntimes: %d' % len(sm.times), LOGFILE)
    ptf('\n\n>> Other model details ', LOGFILE)
    ptf(sm, LOGFILE)

def save_model(sm, RUNID, MODELFILENAME, LOGFILE=None):
    # saves model to file
    model_file = open('./' + RUNID + '/' + MODELFILENAME, 'wb')
    ptf('\n>> Writing model results to %s' % MODELFILENAME, LOGFILE)
    pickle.dump(sm, model_file, -1)
    model_file.close()

if __name__ == '__main__':
    # main(RUNID=RUNID, START_DT_STR=START_DT_STR,
    #     MODELFILENAME=MODELFILENAME,
    #     PICKLE_DATA=PICKLE_DATA,
    #     DO_TESTS=DO_TESTS,
    #     PROFILE=PROFILE,
    #     verbose=verbose,
    #     debug=debug,
    #     RELOAD=RELOAD,
    #     n_cpus=n_cpus,
    #     PICKLE_NAMES=PICKLE_NAMES)
    # runs our job

    RUNID = command_line_process(RUNID)
    # prepare to run job
    LOGFILENAME = 'log_%s_%s.txt' % (RUNID, START_DT_STR)
    LOGFILE = create_logfile(RUNID, LOGFILENAME)

    # get the run conditions for the runid from the json
    # NOTE excludes verbose and debug flags - those are fit parameters
    # and exludes runid since that is set up above
    with open((RUNID + '.json')) as f:
        run_params = json.load(f, object_hook=ascii_encode_dict)

    # to see if more ram is used for more cpus
    n_jobs = run_params['detection_model_arguments']['n_jobs']

    ### Unittests ###
    if DO_TESTS:
        start = time.time()
        ptf( '\n>> Unpickling data ...\n', LOGFILE)
        X = my_unpickle(PICKLE_NAMES[0])
        y = my_unpickle(PICKLE_NAMES[1])
        used_column_headers = my_unpickle(PICKLE_NAMES[2])

        end = time.time()
        ptf( 'Data unpickled in %d seconds (%d total trials)' % ((end-start), len(X)), LOGFILE)

        tsm_unit = run_tsm_unittests(X, y, used_column_headers.values, verbose=verbose, logfile=LOGFILE)
        # sm_unit = run_unittests(X_test, y_test, verbose=False)
    else:
        # ouptput run conditions to screen and logfile
        bigstart = time.time()

        # start memory profiling
        if PROFILE:
            tr, tr_sm = start_memory_profiling


        if RUNTYPE == 'trigger':
            ptf('*** %s - TRIGGERED SERIES MODEL - ***' % RUNID)
        elif RUNTYPE == 'series':
            ptf('*** %s - SERIES MODEL - ***' % RUNID)


        print_job_info(run_params, n_jobs, n_cpus, RUNID, START_DT_STR, LOGFILE=LOGFILE,
            debug=debug, profile=PROFILE, verbose=verbose, start=True)

        if RELOAD:
            X, y, used_column_headers, df, df_raw = reload_data(LOGFILE, PICKLE_DATA)
        else:
            start = time.time()
            ptf( '\n>> Unpickling data ...\n', LOGFILE)
            X = my_unpickle(PICKLE_NAMES[0])
            y = my_unpickle(PICKLE_NAMES[1])
            used_column_headers = my_unpickle(PICKLE_NAMES[2])

            end = time.time()
            ptf( 'Data unpickled in %d seconds (%d total trials)' % ((end-start), len(X)), LOGFILE)

        run_params['logfile'] = LOGFILE
        run_params['runid'] = RUNID

        # create model
        if RUNTYPE == 'trigger':
            sm = TriggeredSeriesModel(used_column_headers.values, **run_params)
        elif RUNTYPE == 'series':
            sm = SeriesModel(**run_params)

        # Altogether now
        print ('** DOING THE FIT **')
        sm.fit(X, y, verbose=verbose, debug=debug)

        bigend = time.time()

        ptf('====> %d seconds (%0.1f mins)' % ((bigend-bigstart), (bigend-bigstart)/60.0), LOGFILE)
        print_job_info(run_params, n_jobs, n_cpus, RUNID, START_DT_STR, LOGFILE=LOGFILE,
            debug=debug, profile=PROFILE, verbose=verbose, start=False)

        print_run_details(X, sm, LOGFILE)

        save_model(sm, RUNID, MODELFILENAME, LOGFILE=LOGFILE)

        ## VIEW RESULTS
        if RUNTYPE == 'trigger':
            make_trigger_plots(sm, y, RUNID, debug=debug)
        elif RUNTYPE == 'series':
            make_series_plots(sm)

        if PROFILE:
            print_memory_profiles(sm, tr, tr_sm, LOGFILE = None)

    LOGFILE.close()
