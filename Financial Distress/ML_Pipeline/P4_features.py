import pandas as pd
import numpy as np
import pylab as pl

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, 
                              GradientBoostingClassifier)
    
def display_importance(df, label, features, method):
    '''
    Given dataframe, label, and list of features,
    plot a graph to rank variable importance
    '''
    
    if method == "Decision Tree":
        model = DecisionTreeClassifier()
    elif method == "Gradient Boosting":
        model = GradientBoostingClassifier()
    elif method == "Random Forest":
        model = RandomForestClassifier()
    else:
        raise ValueError('{} not currently avaliable'.format(method))
        
    model.fit(df[features], df[label])
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    padding = np.arange(len(features)) + 0.5
    plt.barh(padding, importances[sorted_idx], align='center')
    plt.yticks(padding, np.asarray(features)[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance for {}".format(method))
    
def discretize(df, varname, nbins, method='ufov'):
    '''
    Discretizes given "varname" into "nbins".

    Inputs:
            - df: name of pandas DataFrame
            - varname: name of variable to be discretized
            - nbins: number of categories to create
            - method: can be 'quantile', 'uniform', 'linspace' or 'logspace'

    Returns: nothing. Modifies "df" in place
    '''
    
    assert varname in df.columns, "Column '{}' not in DataFrame".format(varname)
    assert len(df[varname].value_counts()) > nbins, "Number of bins too large"

    if method == 'ufov': # Uniform frequency of values
        df[varname + '_cat'] = pd.qcut(df[varname], nbins)
    elif method == 'uv': # Uniform values
        df[varname+'_cat'] = pd.cut(df[varname], nbins)
    elif method == 'linspace':
        minval = min(df[varname])
        maxval = max(df[varname])
        bins = np.linspace(minval, maxval, nbins+1)
        df[varname+'_cat'] = pd.cut(df[varname], bins, include_lowest=True)
    elif method == 'logspace':
        minval = min(df[varname])
        maxval = max(df[varname])
        assert maxval > 0, 'Column {} has only neg or zero numbers'.format(varname)
        if minval <= 0:
            print('Warning, {} has negative or zero values'.format(varname))
            minval = 0.0001
        bins = np.logspace(np.log10(minval), np.log10(maxval), num = nbins + 1)
        df[varname + '_cat'] = pd.cut(df[varname], bins, include_lowest = True)
    else:
        raise ValueError("{} not currently avaliable. Try one of the following: 'ufov','uv','linspace','logspace'.".format(method))
                
def gen_dummies(df, varnames, drop=True):
    '''
    Given a dataframe and certain column, returns a set of dummies
    '''
    for var in varnames:
        for i, value in enumerate(df[var].unique()):
            df[var + '_{}'.format(i+1)] = df[var] == value
        
        if drop:
            df.drop(var, inplace=True, axis=1)
