"""
Module for preprocessing functions
"""
import re
import numpy as np
import pandas as pd

def keep_track_of_missing_values(df, missing_ind):
    """
    Adds columns indicating if original columns had
    missing values.
    In:
        - df: pandas df
        - missing_ind: (str) indicates columns that keep track of
            missing data
    Out:
        - df: function modifies df inplace
        - list of newly created columns
    """
    new_columns = []

    for column in df:

        if df[column].isnull().values.any():
            new_col_name = column + missing_ind
            new_col = np.zeros(len(df), dtype=np.int)
            new_col[df[column].isnull()] = 1
            df[new_col_name] = new_col
            new_columns.append(new_col_name)
    return new_columns


def fill_missing_categorical_values(df, list_of_categorical_vars = []):
    """
    Fills missing categorical values. Missing categorical vars in
    list_of_categorical_vars are replaced by most often occuring category.
    In:
        - df: pandas df
        - list_of_categorical_vars: (list) of categorical variables
    Out:
        - df
    """
    for cat_var in list_of_categorical_vars:
        df[cat_var] = df[cat_var].fillna(df[cat_var].value_counts().idxmax())
    return df


def fill_missing_values(df, class_mean = False, key = None):
    """
    Replaces missing values with mean of class.
    In:
        - df: pandas df
        - class_mean: (bool) whether or not to use mean of class
        - key: column for classes (opt)
    Out:
        - df
    """
    if class_mean:
        key_col = df[key]
        df = df.groupby(key).transform(lambda x: x.fillna(x.mean()))
        df[key] = key_col
        return df
    else:
        for col in df.columns[df.isnull().any()]:
            df[col] = df[col].fillna(df[col].mean())
        return df


def in_bound_test(value, start, end):
    """
    Helper function to recreate discretized dummy
    vars in test set.
    In:
        - value:
        - start:
        - end:
    Out:
        - 1 if value in range; 0 o/w
    """
    if value >= start and value <= end:
        return 1
    else:
        return 0


def insert_discretize_quantiles(df, col_to_value_dict, drop_original=False):
    """
    In:
        - df: pandas dataframe
        - col_to_value_dict: (dict) with columns (keys) to be
            created in df as dummy vars according to values
            {original_col: [(dummy_col, start, end),
                            (dummy_col, start, end)]}
        - drop_original: (bool) whether or not to drop original column
                        that discrete dummies are generated from
    Out:
        - df
    """
    for original_col in col_to_value_dict:

        list_of_dum_cols = col_to_value_dict[original_col]

        for list_of_dum_col in list_of_dum_cols:
            dummy_col, start, end = list_of_dum_col
            df[dummy_col] = df.apply(lambda row: in_bound_test(row[original_col], start, end), axis=1)

        if drop_original:
            del df[original_col]

    return df


def build_col_to_value_dict(df, dummy_code):
    """
    Function that builds dict with discretized columns
    and their dummy columns with cut-off values.
    In:
        - df: pandas dataframe
        - dummy_code: (str) to append to dummy columns
    Out:
        - dict
    """
    col_to_value_dict = {}

    for col in df.columns:
        if dummy_code in col:

            start_pos = col.find(dummy_code + "_")
            original_col = (col[:start_pos])
            if not original_col in col_to_value_dict:
                col_to_value_dict[original_col] = []

            start_pos = col.find(dummy_code + "_")
            cut_off_vals = col[start_pos + len(dummy_code + "_"):]

            start = cut_off_vals[1: cut_off_vals.find(",")]
            if cut_off_vals[0] == "[":
                start = int(start)
            else:
                start = int(start) + 1

            end = cut_off_vals[cut_off_vals.find(",") + 1:]
            numbers = re.findall('[\d\.]+', end)
            if numbers:
                end = float(numbers[0])
            else:
                raise ValueError('Could not find number value for end in cut_off_vals.')
            if cut_off_vals[:-1] == ")":
                end -= 1

            col_to_value_dict[original_col].append((col, start, end))

    return col_to_value_dict


def create_missing_value_colum_in_testset(traindf, testdf, missing_ind):
    """
    Creates same columns indicating missing values as we
    have in training set.
    In:
        - traindf: pandas dataframe with training data
        - testdf: pandas dataframe with test data
        - missing_ind: (str) indicates columns that keep track of
            missing data
    Out:
        - df
    """
    for col in traindf.columns:

        if col[-len(missing_ind):] == missing_ind:

            original_col = col[:col.find(missing_ind)]

            new_col = np.zeros(len(testdf), dtype=np.int)
            new_col[testdf[original_col].isnull()] = 1
            testdf[col] = new_col


def discretize_cont_var(df, cont_var, n, dummy_code, drop=False):
    """
    Discretizes  continuous variable.
    In:
        - df: pandas dataframe
        - cont_var: continues variable to be discretized
        - n: number of percentiles
        - dummy_code: (str) to append to dummy columns
        - drop: (bool) to drop continous variable
    Out:
        - df
    """
    step_size = 1/n
    bucket_array = np.arange(0, 1+step_size, step_size)

    df[cont_var + dummy_code] = pd.qcut(df[cont_var], bucket_array)
    df = pd.get_dummies(df, columns=[cont_var + dummy_code])

    if drop:
        del df[cont_var]

    return df


def dummify_var(df, cat_vars, drop=False):
    """
    Takes categorical variable and creates binary/dummy variables from it.
    In:
        - df: pandas dataframe
        - cat_vars: list of categorical variables
        - drop: (bool) whether or not to drop first dummy
    Out:
        - df: pandas dataframe
    """
    return pd.get_dummies(df, columns=cat_vars, drop_first=drop)
