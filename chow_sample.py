# chow_sample.py
# Anders Berliner
# 20160107

import numpy as np
import pysal
from pysal.spreg.ols_regimes import OLS_Regimes
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import f

#
# db = pysal.open(pysal.examples.get_path('columbus.dbf'),'r')
# y_var = 'CRIME'
# y = np.array([db.by_col(y_var)]).reshape(49,1)
# x_var = ['INC','HOVAL']
# x = np.array([db.by_col(name) for name in x_var]).T
# r_var = 'NSA'
# regimes = np.array(db.by_col(r_var))
# olsr = OLS_Regimes(y, x, regimes, constant_regi='many', nonspat_diag=False, spat_diag=False, name_y=y_var, name_x=x_var, name_ds='columbus', name_regimes=r_var, regime_err_sep=False)
# print olsr.name_x_r #x_var
# print olsr.chow.regi
# print 'Joint test: Wald statistic, p-value'
# print olsr.chow.joint
#
# mask = regimes == 1
# x1 = x[mask]
# x2 = x[~mask]
# y1 = y[mask]
# y2 = y[~mask]

# plt.figure(figsize=(12,6))
# plt.subplot(2,1,1)
# plt.plot(x1[:,0],y1, 'ro', label='1')
# plt.plot(x2[:,0],y2, 'bo', label='0')
# plt.title('Column 0')
# plt.legend()
#
# plt.subplot(2,1,2)
# plt.plot(x1[:,1], y1, 'ro', label='1')
# plt.plot(x2[:,0], y2, 'bo', label='0')
# plt.title('Col 1')
# plt.legend()
# plt.show()

## try my own Data
m1,m2 = 3.099,3.1
b1,b2, = 4.1,4.1
sig1,sig2 = 0.25, 0.33

t = np.arange(0,100, 10)
mask = t>50
y1 = m1*t[mask] + b1 + np.random.randn(len(t[mask]))*sig1
y2 = m2*t[~mask] + b2 + + np.random.randn(len(t[~mask]))*sig2
y = np.hstack((y1,y2)).reshape(-1,1)
t = t.reshape(-1,1)

print 'Trying to find chow values with yi = bi + mi*t + N(0,sigi)'
print '1:',b1,m1,sig1
print '2:',b2,m2,sig2
print len(t), 'total values'

# use the pysal package to find the two regimes
olsr = OLS_Regimes(y, t, mask.astype(int))
print 'OLSR'
print 'beta', olsr.betas
print 'Chow', olsr.chow.joint


def chow_test(x1,y1,x2,y2):
    n1 = len(x1)
    n2 = len(x2)
    k = x1.shape[1]
    df = n1+n2-2*k
    # print x1.shape, x2.shape
    x = np.vstack((x1,x2))
    # print y1.shape, y2.shape
    y = np.vstack((y1.reshape(-1,1),y2.reshape(-1,1)))

    model = LinearRegression()
    model.fit(x1,y1)
    print 'Y1 fit', model.intercept_, model.coef_
    y1p = model.predict(x1)
    s1 = np.sum((y1p-y1)**2)
    # print y1
    # print y1p
    model.fit(x2,y2)
    print 'Y2 fit', model.intercept_, model.coef_
    y2p = model.predict(x2)
    s2 = np.sum((y2p-y2)**2)
    # print y2
    # print y2p
    model.fit(x,y)
    print 'Y fit', model.intercept_, model.coef_
    yp = model.predict(x)
    s = np.sum((yp-y)**2)
    # print y
    # print yp
    print s1,s2,s
    chow = ((s-(s1+s2))/float(k))/((s1+s2)/(n1+n2-2*k))
    print chow

    pval = f.cdf(chow, k, df)
    print 1-pval
    # print f.ppf(chow,k,df)
    return chow


print 'Try on my own'

chow_test(t[mask], y1, t[~mask], y2)
