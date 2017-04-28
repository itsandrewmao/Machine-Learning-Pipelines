# 4. Generate Features/Predictors: For this assignment, you should 
# write one function that can discretize a continuous variable and 
# one function that can take a categorical variable and create binary/dummy 
# variables from it. Apply them to at least one variable each in this data.
    
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pylab as pl

def display_importance(df, label, features):
    '''
    Given dataframe, label, and list of features,
    plot a graph to rank variable importance
    '''
    clf = RandomForestClassifier()
    clf.fit(df[features], df[label])
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    padding = np.arange(len(features)) + 0.5
    pl.barh(padding, importances[sorted_idx], align='center')
    pl.yticks(padding, np.asarray(features)[sorted_idx])
    pl.xlabel("Relative Importance")
    pl.title("Variable Importance")
    
def discretize(df, varname, nbins, cut_type='quantile'):
    '''
    Discretizes given "varname" into "nbins".

    Inputs:
            - df: name of pandas DataFrame
            - varname: name of variable to be discretized
            - nbins: number of categories to create
            - cut_type: can be 'quantile', 'uniform', 'linspace' or 'logspace'

    Returns: nothing. Modifies "df" in place
    '''
    accepted_cut_types = ['quantile', 'uniform', 'linspace', 'logspace']

    assert varname in df.columns, "Column '{}' not found in DataFrame".format(varname)

    assert len(df[varname].value_counts()) > nbins, "Number of bins too large"

    assert cut_type in accepted_cut_types, "Given cut_type not allowed"


    if cut_type == 'quantile':
        df[varname+'_cat'] = pd.qcut(df[varname], nbins)
    elif cut_type == 'uniform':
        df[varname+'_cat'] = pd.cut(df[varname], nbins)
    elif cut_type == 'linspace':
        minval = min(df[varname])
        maxval = max(df[varname])
        bins = np.linspace(minval, maxval, nbins+1)
        df[varname+'_cat'] = pd.cut(df[varname], bins, include_lowest=True)
    elif cut_type == 'logspace':
        minval = min(df[varname])
        maxval = max(df[varname])
        
        assert maxval > 0, 'Column {} has only negative or zero numbers'.format(varname)

        if minval <= 0:
            print('Warning, {} has negative or zero values'.format(varname))
            minval = 0.0001

        bins = np.logspace(np.log10(minval), np.log10(maxval), num = nbins+1)
        df[varname+'_cat'] = pd.cut(df[varname], bins, include_lowest=True)


def gen_dummies(df, col):
    '''
    Given a dataframe and certain column, returns a set of dummies
    '''
    for i, value in enumerate(df[col].unique()):
        df[col + '_{}'.format(i)] = df[col] == value
