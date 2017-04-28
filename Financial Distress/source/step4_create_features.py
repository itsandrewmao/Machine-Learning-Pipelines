import pandas as pd
import numpy as np
# 4. Generate Features/Predictors: For this assignment, you should write one 
# function that can discretize a continuous variable and one function that can 
# take a categorical variable and create binary/dummy variables from it. Apply 
# them to at least one variable each in this data.

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


def make_dummies(df, varname):
    '''
    Creates a set of dummies out of "varname"
    '''
    for i, value in enumerate(df[varname].unique()):
        print(i, type(value), value)
        df[varname+'_{}'.format(i)] = df[varname] == value
