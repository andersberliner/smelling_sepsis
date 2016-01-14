# capstone_r_tools.py
# Anders Berliner
# 20160112
#from rpy2.robjects import pandas2ri, r
import pandas as pd
import numpy as np
import time
import pickle

def make_long_dataframe(X, y, column_headers, result_type='detection', result_value=0, Ntimes=10):
    t1 = time.time()

    Ncolors = 3
    Nsamp = len(X)

    col_nums = list(set([int(x[:-1]) for x in column_headers[1:]]))
    Nspots = len(col_nums)

    tidxs = y.index.values

    # Convert series to frame
    Z = X.to_frame()

    # flatten arrays in data frame
    start = time.time()
    print 'Flattening'
    final = np.array(Z['data'].iloc[0])
    for i in xrange(1, Nsamp):
        final = np.vstack((final, Z['data'].iloc[i]))
    # print 'NP-2D', final.shape

    end = time.time()
    print 'Flattened %d trials in %d secs' % (Nsamp, (end-start))

    # omit first row which is time
    nplong = final[:,1:].flatten()
    # print 'NPlong', nplong.shape

    # data column
    start = time.time()
    print 'Raveling...'
    nplong = nplong.ravel()
    # print 'NPlong-1D', nplong.shape
    longdf = pd.DataFrame({'DI':nplong})
    # print longdf.head()

    end = time.time()
    print 'Raveled %d trial in %d secs' % (Nsamp, (end-start))

    # color column - Repeat for Nspots*Ntimes*Nsamp
    print 'add color'
    longdf['color'] = ['R', 'G', 'B']*(Nspots*Ntimes*Nsamp)
    # print longdf['color'].unique()
    # print longdf.head()

    # spot column - separated by Ncolors - Repeat for Ntimes*Nsamp
    # spots go [2,2,2]
    print 'add spot'
    spots = np.array([col_nums[x]*np.ones(Ncolors) for x in range(Nspots)]).flatten()
    spots = list(spots)*(Ntimes*Nsamp)
    longdf['spot'] = spots
    # print longdf['spot'].unique()
    # print longdf.head()

    # time column
    print 'add time'
    times = np.array([20*x*np.ones(Ncolors*Nspots) for x in range(Ntimes)]).flatten()
    times = list(times)*(Nsamp)
    longdf['time'] = times
    # print longdf['time'].unique()
    # print longdf.head()

    # trial columns
    print 'add trial'
    trials = np.array([tidxs[x]*np.ones(Ncolors*Nspots*Ntimes) for x in range(Nsamp)]).flatten()
    trials = list(trials)
    longdf['trial'] = trials
    # print longdf['trial'].unique()
    # print longdf.head()

    # add labels
    print 'add labels'
    detections = np.array([[y.ix[tidxs[x]]['detection']]*(Ncolors*Nspots*Ntimes) for x in range(Nsamp)]).flatten()
    detections = list(detections)
    longdf['detection'] = detections

    grams = np.array([[y.ix[tidxs[x]]['gram']]*(Ncolors*Nspots*Ntimes) for x in range(Nsamp)]).flatten()
    grams = list(grams)
    longdf['gram'] = grams

    classifications = np.array([[y.ix[tidxs[x]]['classification']]*(Ncolors*Nspots*Ntimes) for x in range(Nsamp)]).flatten()
    classifications = list(classifications)
    longdf['classification'] = classifications

    # color column
    print 'color dummies'
    longdf = pd.get_dummies(longdf, columns=['color'])
    longdf.drop('color_B', axis=1, inplace=True)
    # print longdf.head()

    # label dummies
    print 'label dummies'
    longdf = pd.get_dummies(longdf, columns=['gram', 'classification'])
    # don't need to drop dummies
    # longdf.drop(['gram_Control', 'classification_Control'], axis=1, inplace=True)

    # pickle
    start = time.time()
    print 'Pickling...'
    myfile = open('long_%d.pkl' % Ntimes, 'wb')
    pickle.dump(longdf, myfile, -1)
    myfile.close()

    end = time.time()
    print 'Pickled %d trial in %d secs' % (Nsamp, (end-start))

    # save to R
    start = time.time()
    print 'Zipping...'
    pandas2ri.activate()
    rX = pandas2ri.py2ri(longdf)
    r.assign('hugedata', rX)
    r("save(longdata, file='longdata.gzip', compress=TRUE)")
    pandas2ri.deactivate()

    end = time.time()
    print 'Zipped %d trial in %d secs' % (Nsamp, (end-start))

    t2 = time.time()
    print 'Total time %d secs' % (t2-t1)

    return longdf
