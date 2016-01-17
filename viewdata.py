# viewdata.py

import numpy as np
import pandas as pd
import pickle
import time
from seriesmodel import SeriesModel
from featurizer import PolynomialFeaturizer
from timeseriesplotter import SpotTimePlot

myfile = open('pickles/DI.pkl', 'rb')
X = pickle.load(myfile)
myfile.close()
# print len(X), X.iloc[0].shape

myfile = open('pickles/used_column_headers.pkl', 'rb')
column_headers = pickle.load(myfile)
myfile.close()
# print len(column_headers)

myfile = open('pickles/ydf.pkl', 'rb')
y = pickle.load(myfile)
myfile.close()
# print y.head()




# sm = SeriesModel(reference_time=9, min_time=3)
# PF = PolynomialFeaturizer(n=4, reference_time=9, verbose=True)
# mycoefs, myscores = PF.fit_transform(X)
#
#
# DI_pred = PF.predict(X, mycoefs)
STP = SpotTimePlot(y, column_headers)
# STP.plot_raws(X, averages=False)
STP.plot_raws(X)
