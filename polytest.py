# polytest.py

# test every part of my algorithms
# make some fake data
import pandas as pd
import numpy as np
from seriesmodel import SeriesModel
from featurizer import PolynomialFeaturizer
from timeseriesplotter import SpotTimePlot
import matplotlib.pyplot as plt
from itertools import izip
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LassoCV, RidgeCV

#  Create dummy classes that are just lines and should be easily seperable.
def f1(t):
    x = 2* t + 1 + 2*np.random.randn((len(t)))
    return x

def f2(t):
    x = 3*t - 2 + 2*np.random.randn((len(t)))
    return x

def add_t1(df, t):
    data = np.hstack((t.reshape(-1,1), f1(t).reshape(-1,1), f2(t).reshape(-1,1)))
    return df.append({'data':data, 'classification':'1', 'gram':'1', 'detection':1}, ignore_index=True)
def add_t2(df, t):
    data = np.hstack((t.reshape(-1,1), f2(t).reshape(-1,1), f1(t).reshape(-1,1)))
    return df.append({'data':data, 'classification':'2', 'gram':'1', 'detection':1}, ignore_index=True)
def add_t3(df, t):
    data = np.hstack((t.reshape(-1,1), 0.5*f2(t).reshape(-1,1), 0.05*f2(t).reshape(-1,1)))
    return df.append({'data':data, 'classification':'Control', 'gram':'Control', 'detection':0}, ignore_index=True)
def add_t4(df, t):
    data = np.hstack((t.reshape(-1,1), 3*f1(t).reshape(-1,1), f2(t).reshape(-1,1)))
    return df.append({'data':data, 'classification':'3', 'gram':'2', 'detection':1}, ignore_index=True)
def add_t5(df, t):
    data = np.hstack((t.reshape(-1,1), -1*f1(t).reshape(-1,1), f2(t).reshape(-1,1)))
    return df.append({'data':data, 'classification':'4', 'gram':'2', 'detection':1}, ignore_index=True)

if __name__ == '__main__':
    print 'MAKE DATAFRAME'
    df = pd.DataFrame(columns=['data','classification', 'gram', 'detection'])
    t = np.arange(0,25,1)
    for i in range(0,10):
        df = add_t1(df,t)
        df = add_t5(df,t)
        df = add_t4(df,t)
    for i in range(0,5):
        df = add_t3(df,t)
    for i in range(0,8):
        df = add_t2(df,t)
    print df.head()

    print 'MAKE X,Y'
    X = df['data']
    y = df.drop(['data'], axis=1)
    print y.head()
    print X.head()

    print 'Fit a  line with OLS'
    ## inspect the output for what we would expect ##

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=4)

    T = poly.fit_transform(t.reshape(-1,1))
    # print T
    x_line = f2(t)
    print T[0:4]
    print x_line
    plt.figure(figsize=(15,10))
    models = [LinearRegression(fit_intercept=False), Lasso(fit_intercept=False), Ridge(fit_intercept=False),
        LassoCV(fit_intercept=False), RidgeCV(fit_intercept=False)]
    names = ['ols', 'lasso', 'ridge', 'lassocv', 'ridgecv']
    for model, name in izip(models, names):
        model.fit(T, x_line)
        print '%s Regressed to:' % name
        print model.coef_

        plt.plot(t, np.dot(T, model.coef_), label=name)

    plt.plot(t, x_line, 'ko', label='raw')
    plt.title('Fitting 3t-2 with a 4th order polynomial')
    plt.legend()
    plt.ylim([-5,95])
    # plt.show()

    print 'Now try with my featurizer'

    newdf = pd.DataFrame(columns = ['data'])

    def make_data(t):
        Z = np.zeros((len(t), 3))
        Z[:,0] = t.copy()
        Z[:,1] = f2(t)
        Z[:,2] = f1(t)
        return Z

    for i in range(14):
        newdf = newdf.append({'data': make_data(t)}, ignore_index=True)
    print newdf.head()
    print newdf.iloc[1]['data']

    poly = PolynomialFeaturizer(n=4)

    Xf, scores = poly.fit_transform(newdf['data'])

    print newdf.iloc[0]['data'][0:5]
    print Xf.iloc[0]
    print Xf.iloc[1]

    plt.figure(figsize=(15,10))
    plt.plot(t, newdf.iloc[0]['data'][:,1], 'ko', label='raw')
    plt.plot(t, np.dot(T, Xf.iloc[0][:,0]), label='fit with PolynomialFeaturizer')
    plt.legend()
    plt.title('Using PF to fit with 4th order: %s' % Xf.iloc[0][:,0])
    plt.ylim([-5,95])
    # plt.show()

    print 'Now try with series model'
    Xnew = newdf['data']
    ynew = pd.DataFrame({'detection': [0,1,1,1,0,1,1,0,1,1,1,0,1,1],
        'gram': ['Control', 'p', 'p', 'p', 'Control', 'n', 'n','Control', 'p', 'p', 'p', 'Control', 'n', 'n'],
        'classification': ['Control', 'A', 'A', 'A', 'Control', 'A', 'A','Control', 'A', 'A', 'A', 'Control', 'A', 'A']})
    smf = SeriesModel(max_time=25, on_disk=False, nfolds=1, fold_size=0.10, detection_featurizer='poly',
        detection_featurizer_arguments={'n': 4}, detection_scaler=None, detection_reducer=None,
        reference_time = 0, color_vector_type='I')
    smf.number_of_columns = 3
    smf.fit(Xnew, ynew)

    print smf.features['detection'][24][0][0:5]
    a = Xnew.iloc[0][:,0]
    b = Xnew.iloc[0][:,1]
    c = smf.features['detection'][24][0][0:5]
    print c
    print smf.make_rect_features(smf.features['detection'][24][0], 4+1, Xnew.iloc[0].shape[1]-1)
    d = np.dot(T, c)

    plt.figure(figsize=(15,10))
    plt.plot(a,b,'ko',label='raw')
    plt.plot(t,d,label='smf')
    plt.legend()
    plt.ylim([-5,95])
    plt.title('Series Model fit %s' % c)
    # plt.show()

    print 'And now with a on_disk series model'
    smfod = SeriesModel(max_time=25, on_disk=True, nfolds=1, fold_size=0.10, detection_featurizer='poly',
        detection_featurizer_arguments={'n': 4}, detection_scaler=None, detection_reducer=None,
        reference_time = 0, color_vector_type='I')
    smfod.number_of_columns = 3
    smfod.fit(Xnew, ynew)

    c = smfod.load_time_step('features', t=24)['detection'][0]
    a = Xnew.iloc[0][:,0]
    b = Xnew.iloc[0][:,1]
    # c = smfod.features['detection'][24][0][0:5]
    print c
    print smfod.make_rect_features(c, 4+1, Xnew.iloc[0].shape[1]-1)
    d = np.dot(T, c[0:5])

    plt.figure(figsize=(15,10))
    plt.plot(a,b,'ko',label='raw')
    plt.plot(t,d,label='smf')
    plt.legend()
    plt.ylim([-5,95])
    plt.title('Series Model fit on disk %s' % c)


    plt.show()
