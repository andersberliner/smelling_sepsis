# math_capstone.py
# Anders Berliner
import numpy as np

def tridiag(diagonal_minus1, diagonal, diagonal_plus1, k1=-1, k2=0, k3=1):
#     http://stackoverflow.com/questions/5842903/block-tridiagonal-matrix-python
    return np.diag(diagonal_minus1, k1) + np.diag(diagonal, k2) + np.diag(diagonal_plus1, k3)

def make_A(f):
    # build A
    d = 4*np.ones(len(f))
    dm1 = np.ones(len(f)-1)
    dp1 = np.ones(len(f)-1)
    # centralized difference at most points
    A = tridiag(dm1, d, dp1)
    # forward difference at first point
    A[0,0] = 1
    A[0,1] = 2
    # backward difference at last point
    A[-1,-2] = 2
    A[-1,-1] = 1
    return A

def make_b(f):
    # Construct b[i] = 3*(f[i+1] - f[i-1])
    # Except b[0] = -2.5*f0 + 2*f1 + 0.5*f2
    # b[n] = 2.5*fn - 2*fn-1 - 0.5fn-2
    # print 'f', f.shape
    # remove first value from f and slide all values left
    fp1 = np.delete(f,0,axis=0)
    # print fp1.shape
    # pad last value of f plus 1
    fpp1 = np.vstack((fp1, [[0]]))
    # remove last value from f
    fm1 = np.delete(f, -1, axis=0)
    # pad first value of f minus 1
    fmm1 = np.vstack(([[0]],fm1))
    # set appropriate values of pade
    b = 3*(fpp1-fmm1)
    b[0] = -2.5*f[0] + 2*f[1] + 0.5*f[2]
    b[-1] = 2.5*f[-1] - 2*f[-2] - 0.5*f[-3]
    return b

def pade(f, h=1.0):
    '''
    Pade approximation of the numerical derivative for EVENLY SPACED data
        Solves A*fp = 1/h*b
        Where A is a sparse matrix
        b is the values at various point
        h is the (uniform) step
        fp is numerical derivative at a point

    IN:
        f - np array - a N X 1 numpy array of f(x) values (assumed to be) evaluated at
            evenly space x
        h - float - spacing between points
    OUT:
        fp - np array - a N X 1 numpy array of the numerical derivatives f'(x)
        score - float - sum of the residuals from the np.linalg.lstsq call
    '''
    if len(f) <=3:
        print 'ERR: need more points to use pade'
    f = f.reshape(-1,1)
    # print 'pade', f.shape
    A = make_A(f)
    # print A.shape
    b = (1/h)*make_b(f)
    fp = np.linalg.lstsq(A, b)
    return fp[0], fp[1]

def my_sigmoid(t, k=1, A=1, B=0):
    y = 1/(A + np.exp(-(k*t + B)))
    return y

def my_sigmoid_prime(t,k=1,A=1,B=0):
    y = np.exp(-B-k*t)*k/(A+np.exp(-B-k*t))**2
    return y

def my_sigmoid_prime_prime(t,k=1,A=1,B=0):
    y1 = np.exp(-2*B-2*k*t)*k*k/(A+np.exp(-B-k*t))**3
    y2 = np.exp(-B-k*t)*k*k/(A+np.exp(-B-k*t))**2
    return 2*y1-y2
