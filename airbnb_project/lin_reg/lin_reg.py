import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100, reg=0):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate
    iterations = no of iterations

    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    # print(m)
    # print('iteration number: ',end = " ")
    for it in range(iterations):
        prediction = np.dot(X, theta)
        # print("pred shape",prediction.shape)
        # print("Y shape", y.shape)
        d_score = prediction - y
        # print("dscore shape",d_score.shape)
        if it % int(0.1 * iterations) == 0:
            print(it, end = ', ')
        theta = theta - (1 / m) * learning_rate * (X.T.dot(d_score) - reg * np.sum(np.square(theta)))

    return theta


