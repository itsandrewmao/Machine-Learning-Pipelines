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


def fill_in(df, var, d, method):
    '''
    Methods (str) refer to filling in missing values with
    "mean", "forward"(filling), or "backward"(filling)
    
    *Refer to diff_in_mean function in describe.py
    for requirements of var_lst and d.
    '''
    assert method in {"mean", "forward", "backward"}, "Only supports mean/ffill/bfill"
    name = d['x'+str(var)]
    if method == 'mean':
        df[name] = df[name].fillna(df[name].mean())
    if method == 'forward':
        df[name] = df[name].ffill()
    else:
        df[name] = df[name].bfill()