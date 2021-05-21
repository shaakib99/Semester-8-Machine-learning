import numpy as np
from computeCost import *


def gradient_descent(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        J_history[i] = compute_cost(X, y, theta)
        for j in range(len(theta)):
            theta[j] = theta[j] - (alpha*(1/m)*subtract(X,y,theta,m,j))

    return theta, J_history


def subtract(X,y,theta,m,t):
    sub = 0
    for i in range(m):
        sub +=  ((theta[0] + np.dot(theta[1],X[i][1])) - y[i])*X[i][t]
    return  sub