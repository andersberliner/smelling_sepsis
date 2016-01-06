# rpy2_tutorial.py
# Anders Berliner
# 20160105

# import rpy2.robjects as robjects
# # R package names
# packnames = ('ggplot2', 'hexbin')
#
# # import rpy2's package module
# import rpy2.robjects.packages as rpackages
#
# if all(rpackages.isinstalled(x) for x in packnames):
#     have_tutorial_packages = True
# else:
#     have_tutorial_packages = False
#
# if not have_tutorial_packages:
#     # import R's utility package
#     utils = rpackages.importr('utils')
#     # select a mirror for R packages
#     utils.chooseCRANmirror(ind=1) # select the first mirror in the list
#
# if not have_tutorial_packages:
#     # R vector of strings
#     from rpy2.robjects.vectors import StrVector
#     # file
#     packnames_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
#     if len(packnames_to_install) > 0:
#         utils.install_packages(StrVector(packnames_to_install))
#
#
# ## numpy to rpy2 and vice-versa
# import numpy as np
# from rpy2.robjects.packages import importr, data
#
# ltr = robjects.r.letters
# ltr_np = np.array(ltr)
#
#
# datasets = importr('datasets')
# ostatus = data(datasets).fetch('occupationalStatus')['occupationalStatus']
# ostatus_np = np.array(ostatus)
# ostatus_npnc = np.asarray(ostatus)

from rpy2.robjects import numpy2ri
numpy2ri.activate()
numpy2ri.deactivate()

# let's generate and fit some random data using python
print 'Creating some data to try fitting via OLS LR'
import numpy as np
X = np.arange(1,10)
# X = X.reshape(-1,1)
y = X + np.random.randn(9)*0.2
X = X.reshape(-1,1)
print 'X: ', X
print 'y: ', y

print 'Fit with sklearn...'

from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import LinearRegression
# model = OLS(y, X)
# results = model.fit()
# print results.summary()
model = LinearRegression()
model.fit(X,y)



print 'Try to fit with r stuff..'
from rpy2.robjects.packages import importr
from rpy2.robjects import Environment, Formula
numpy2ri.activate()
fmla = Formula('y ~ x')
model_env = fmla.environment

stats = importr('stats')
# model_env['fmla'] = Formula('y ~ x')
model_env['x'] = X
model_env['y'] = y

fit = stats.lm(fmla)

# base = importr('base')
# fit = base.eval.rcall(base.parse(text = 'lm(fmla)'), stats._env, model_env)
# fit = base.eval()

print 'Stuff available to us after the fit'
print fit.names

print 'Compare the fits...'
print '\t|sklearn\t|r'
print 'Intercept\t%f\t%f' % (model.intercept_,fit.rx2('coefficients')[0])
print 'Slope\t%f\t%f' % ( model.coef_, fit.rx2('coefficients')[1])
print 'R2\t%f\t%s' % (model.score(X,y), '?')
