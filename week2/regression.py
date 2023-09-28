import pandas as pd 
import numpy as np

def train_linear_regression(X_train: np.array, y_train):
    # fit on train
    '''
    Parameters:
        X_train: 2-D array of features
        y_train: 1-D array of target variable
    The function calculates weights for linear regression equation.
    Returns:
        w[0] -> float, bias (y-intersect)
        w[1:] -> array of weights (floats)
    '''
    # add 1 to the beginning of every vector in features
    X = np.insert(X_train, 0, np.ones(len(X_train)), axis = 1)
    # get gram matrix
    XTX = X.T.dot(X)
    # inverse XTX
    XTX_inv = np.linalg.inv(XTX)
    # calculate weights
    w = XTX_inv.dot(X.T).dot(y_train)
    bias = w[0]
    weights = w[1:]

    return bias, weights

def train_linear_regression_reg(X_train: np.array, y_train: np.array, r:int=0.01):
    # fit on train
    # added regularization
    '''
    Parameters:
        X_train: 2-D array of features
        y_train: 1-D array of target variable
    The function calculates weights for linear regression equation.
    Returns:
        w[0] -> float, bias (y-intersect)
        w[1:] -> array of weights (floats)
    '''
    # add 1 to the beginning of every vector in features
    X = np.insert(X_train, 0, np.ones(len(X_train)), axis = 1)
    # get gram matrix
    XTX = X.T.dot(X)
    # regularization
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg
    # inverse XTX
    XTX_inv = np.linalg.inv(XTX)
    # calculate weights
    w = XTX_inv.dot(X.T).dot(y_train)
    bias = w[0]
    weights = w[1:]

    return bias, weights

def predict_y(X: np.array, bias, weights):
    return bias + X.dot(weights)

def rmse(y, y_pred):
    ''' 
    y - actual prices
    y_pred - predicted prices

    calculates RMSE score
    '''
    error = y - y_pred
    mse = (error ** 2).mean()
    return np.sqrt(mse)

