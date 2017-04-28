import pandas as pd
# 3. Pre-Process Data: For this assignment, you can limit this to filling in 
# missing values for the variables that have missing values. You can use any 
# simple method to do it (use mean to fill in missing values).

def fill_miss(df, varname, method='mean'):
    
    missing_df = df[df[varname].isnull()].copy()
    not_missing_df = df[~df[varname].isnull()].copy()

    if method == 'mean':
        fill_value = df[varname].mean()
    elif method == 'median':
        fill_value = df[varname].median()

    missing_df.loc[:, (varname)] = fill_value

    not_missing_df.append(missing_df)

    return not_missing_df

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
