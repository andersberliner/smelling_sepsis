import numpy as np
from regression_functions import add_intercept
# slight modifications from Galvanize solutions


class GradientDescent(object):

    def __init__(self, cost, gradient, predict_func, fit_intercept=True, scale=False):
        '''
        INPUT: GradientDescent, function, function
        OUTPUT: None
        Initialize class variables. Takes two functions:
        cost: the cost function to be minimized
        gradient: function to calculate the gradient of the cost function
        '''
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.scale = scale
        self.mu = None
        self.sigma = None
        self.fit_intercept = fit_intercept

    def run(self, X, y, alpha=0.01, num_iterations=10000, step_size=None):
        '''
        INPUT: GradientDescent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None
        Run the gradient ascent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''
        ## Calculate mu and sigma for normalization
        self.calculate_scaling_factors()
        ## Scale X and add intercept (if we want to)
        X = self._maybe_modify_matrix(X)

        ## Initialize coeffs to all zeros
        self.coeffs = np.zeros(X.shape[1])

        if step_size is None:
            for i in xrange(num_iterations):
                self.coeffs += alpha / X.shape[0] * self.gradient(X, y, self.coeffs)
        else:
            diff = step_size
            old_cost = 100000.
            while diff >= step_size:
                total_cost = 0
                # CHANGE HERE FROM SOLN on the SIGN
                self.coeffs -= alpha / X.shape[0] * self.gradient(X, y, self.coeffs)
                total_cost += self.cost(X, y, self.coeffs)
                diff = np.abs((total_cost - old_cost) / old_cost)
                old_cost = total_cost

    def sgd_run(self, X, y, alpha=0.01, step_size=.0001):
        '''
        INPUT: GradientDescent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None
        Run the gradient ascent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''
        ## Calculate mu and sigma for normalization
        self.calculate_scaling_factors()
        ## Scale X and add intercept (if we want to)
        X = self._maybe_modify_matrix(X)

        ## Initialize coeffs to all zeros
        self.coeffs = np.zeros(X.shape[1])

        diff = step_size
        old_cost = 100000.
        while diff >= step_size:
            # shuffle
            indices = np.array(range(0,len(X)))
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]
            # iterate over dataset and update coefficients
            total_cost = 0
            for index in range(X.shape[0]):
                x_i = np.array([X[index,:]]) # wrapped row in array( ) so that the same gradient and cost functions can be used
                y_i = np.array([y[index]])
                # CHANGE HERE FROM THE SOLNS IN THE SIGN
                self.coeffs -= alpha / X.shape[0] * self.gradient(x_i, y_i, self.coeffs)
            total_cost += self.cost(X, y, self.coeffs)
            diff = np.abs((total_cost - old_cost) / old_cost)
            old_cost = total_cost

    def predict(self, X):
        '''
        INPUT: GradientDescent, 2 dimensional numpy array
        OUTPUT: numpy array (ints)
        Use the coeffs to compute the prediction for X. Return an array of 0's
        and 1's.
        '''
        X = self._maybe_modify_matrix(X)
        return self.predict_func(X, self.coeffs)

    def calculate_scaling_factors(self):
        if not self.scale:
            return
        self.mu = np.average(X, axis=0)
        self.sigma = np.std(X, axis=0)

    def _maybe_modify_matrix(self, X):
        if self.scale:
            X = (X - self.mu) / self.sigma
        if self.fit_intercept:
            X = add_intercept(X)
        return X
