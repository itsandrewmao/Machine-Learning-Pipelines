import numpy as np
import pandas as pd

def check_missing_value(df):
    '''
    Prints out the variables that
    have missing values.
    -> so that user can determine which
    method to adopt for filling missing values.
    '''
    n = len(df.index)
    for item in df.columns:
        if df[item].count() < n:
            print(item+" has missing values.")


def fill_miss(df, var, d, method):
    '''
    Fill in missing values for a given column in a dataframe
    
    Given a dataframe, specific column, and fill value method, 
    return the same dataframe without missing values.
    '''
    
    name = d['x'+str(var)]
    
    if method == 'mean':
        df[name] = df[name].fillna(df[name].mean())
    elif method == 'median':
        df[col] = df[col].fillna(df[col].median())
    elif method == 'drop' or method == 'zero':
        df[col] = df[col].fillna(0)
    elif method == "ffill" or method == 'forward':
        df[name] = df[name].ffill()
    elif method == "bfill" or method == 'backward':
        df[name] = df[name].ffill()
    else:
        df[name] = df[name].bfill()
        
def convert_column_type(df, varname, to_type, value_if_true = None):
    
    if to_type == 'bool':
        if len(df[varname].value_counts()) > 2:
            print('Warning, {} has more than 2 values'.format(varname))
    
        df[varname] = df[varname] == 1

    elif to_type == 'int':
        missing_df = df[df[varname].isnull()][varname].copy()
        not_missing_df = df[~df[varname].isnull()][varname].copy()
        not_missing_df = not_missing_df.astype(int)

        not_missing_df.append(missing_df)

        df[varname] = not_missing_df