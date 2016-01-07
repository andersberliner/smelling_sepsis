import numpy as np
# Based on galvanize sprint implementation


def cost(y_pred, y, coeffs, l=0):
    '''
    INPUT: numpy array, numpy array, numpy array, numpy array, float
    OUTPUT: float
    Calculates the RSS with the given coefficients.
    lambda is the regularization parameter.
    '''
    # Because there was random noise added to the y we fit
    # the max value of y can be above 1/A
    # this leads to undefined logit(y,A) = ln(Ay/(1-Ay))
    # When calculating residuals of the logit, I ignore the nan points
    mask = ~np.isnan(y)
    y = y[mask]
    y_pred = y_pred[mask]

    RSS = np.sum((y-y_pred)**2)
    if l:
        return RSS - l * np.sum(coeffs ** 2)
    return RSS


def hypothesis(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array
    Calculate the predicted percentages (floats between 0 and 1) for the given
    data with the given coefficients.
    '''
    return 1.0 / (1.0 + np.exp(- np.dot(X, coeffs)))

def predict(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array
    Calculate the predicted values (0 or 1) for the given data with the given
    coefficients.
    '''
    return np.around(hypothesis(X, coeffs)).astype(int)

def log_likelihood(X, y, coeffs, l=0):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array, float
    OUTPUT: float
    Calculate the log likelihood of the data with the given coefficients.
    lambda is the regularization parameter.
    '''
    y_pred = hypothesis(X, coeffs)
    likelihood = y.dot(np.log(y_pred)) + (1 - y).dot(np.log(1 - y_pred))
    if l:
        return likelihood - l * np.sum(coeffs ** 2)
    return likelihood

def log_likelihood_gradient(X, y, coeffs, l=0):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array, float
    OUTPUT: numpy array
    Calculate the gradient of the log likelihood at the given value for the
    coeffs. Return an array of the same size as the coeffs array.
    lambda is the regularization parameter.
    '''
    gradient = X.T.dot(y - hypothesis(X, coeffs))
    if l:
        return gradient - 2 * l * coeffs
    return gradient

def add_intercept(X):
    '''
    INPUT: 2 dimensional numpy array
    OUTPUT: 2 dimensional numpy array
    Return a new 2d array with a column of ones added as the first
    column of X.
    '''
    return np.hstack((np.ones((X.shape[0], 1)), X))

def accuracy(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUPUT: float
    Calculate the percent of predictions which equal the true values.
    '''
    if y_true.shape != y_pred.shape:
        raise Exception("arrays different sizes")
    return float(np.sum(y_true == y_pred)) / len(y_true)

def precision(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: float
    Calculate the percent of positive predictions which were correct.
    '''
    if y_true.shape != y_pred.shape:
        raise Exception("arrays different sizes")
    return float(np.sum(y_true * y_pred)) / np.sum(y_pred)

def recall(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: float
    Calculate the percent of positive cases which were correctly predicted.
    '''
    if y_true.shape != y_pred.shape:
        raise Exception("arrays different sizes")
    return float(np.sum(y_true * y_pred)) / np.sum(y_true)
