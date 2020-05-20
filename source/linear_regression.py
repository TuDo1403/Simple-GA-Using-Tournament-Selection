import numpy as np
from numpy import transpose
from numpy.linalg import inv

def compute_cost(X, y, theta):
    m = len(y)
    cost = 0.5/m * np.sum((X.dot(theta) - y)**2)
    return cost

def gradient_descend(X, y, theta, alpha, iterations):
    def delta():
        m = len(y)
        delta = 1/m * transpose(X.dot(theta) - y).dot(X)
        return np.around(delta, 5) 
    
    for i in range(iterations):
        temp = theta - (alpha * transpose(delta()))
        theta = np.round_(temp, 2)

    return theta

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.max(X, axis=0) - np.min(X, axis=0)
    return np.divide((X - mu), sigma)

def normal_equation(X, y):
    return inv(transpose(X).dot(X)).dot(transpose(X)).dot(y)
