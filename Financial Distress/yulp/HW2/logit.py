import pandas as pd
from sklearn.linear_model import LogisticRegression

def all_data(df, var_lst, d):
    '''
    Determine which independent variables
    to be used in the logit regression.

    Return dataframes of all available data
    for y and x's.
    '''
    Y = df[d['y']]
    ind_vars = []
    for i in var_lst:
        ind_vars.append(d['x'+str(i)])
    X = df[ind_vars]
    return X, Y

def logit_regression(X_train, Y_train):
    '''
    Fits training data into logit model.
    '''
    model = LogisticRegression()
    model = model.fit(X_train, Y_train)
    return model