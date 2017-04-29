import numpy as np
import pandas as pd

def check_miss(df):
    '''
    Prints out the variables that
    have missing values.
    -> so that user can determine which
    method to adopt for filling missing values.
    '''
    for varname in df.columns:
        if any(df[varname].isnull()) == True:
            print(varname + " has missing values!")

def fill_miss(df, varname, method):
    '''
    Fill in missing values for a given column in a dataframe
    
    Given a dataframe, specific column, and fill value method, 
    return the same dataframe without missing values.
    '''
    
    if method == 'mean':
        df[varname] = df[varname].fillna(df[varname].mean())
    elif method == 'median':
        df[varname] = df[varname].fillna(df[varname].median())
    elif method == 'drop' or method == 'zero':
        df[varname] = df[varname].fillna(0)
    elif method == "ffill" or method == 'forward':
        df[varname] = df[varname].ffill()
    elif method == "bfill" or method == 'backward':
        df[varname] = df[varname].ffill()
    else:
        raise ValueError('{} not currently avaliable'.format(method))
        
def convert_vartype(df, varname, method):
    
    if method == 'bool':
        if len(df[varname].value_counts()) > 2:
            print('Warning, {} has more than 2 values'.format(varname))
    
        df[varname] = df[varname] == 1 

    elif method == 'int':
        pd.options.display.float_format = '{:,.0f}'.format
        
    else:
        raise ValueError('{} not currently avaliable'.format(method))