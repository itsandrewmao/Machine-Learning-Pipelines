import pandas as pd

def read(file_name):
    '''
    Read file into pandas dataframe 
    
    Given file_name of one of the formats below, return the data in a df
    '''
    
    if '.csv' in file_name:
        df = pd.read_csv(file_name)
    elif '.dta' in file_name:
        df = pd.read_stata(file_name)
    elif '.xls' in file_name:
        df = pd.read_excel(file_name)
    elif '.json' in file_name:
        df = pd.read_json(file_name)
    elif '.sql' in file_name:
        df = pd.read_sql(file_name)
    else:
        raise ValueError('{file} is an unsupported file format.'.format(file=file_name))

    return df      