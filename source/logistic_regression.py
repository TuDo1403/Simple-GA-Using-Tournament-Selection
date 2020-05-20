import scipy.optimize as op
import numpy as np
from numpy import log

def sigmoid_func(z):
    g = 1 / (1 + np.exp(-z))
    return np.reshape(g, (len(g), 1))

def cost_function(theta, X, y):
    h = sigmoid_func(X.dot(theta))
    m = len(y)

    cost = -1/m * (y.T.dot(log(h)) + (1-y).T.dot(log(1-h)))
    return cost

def gradient(theta, X, y):
    h = sigmoid_func(X.dot(theta))
    m = len(y)

    gradient = 1/m * ((h-y).T.dot(X)).T
    return gradient.flatten()

def fminunc(initial_theta, X, y):
    result = op.minimize(fun=cost_function, x0=initial_theta, args=(X, y), method='BFGS', jac=gradient)
    optimal_theta = result.x
    return optimal_theta 