import pandas as pd
import re

def read_file(filename):
    '''
    Read a csv or excel file into a pandas DataFrame.
    '''
    ext = re.search('.\w+$', filename).group()
    assert ext in {'.csv', '.xls', '.xlsx'}, "Only takes .csv/.xls/.xlsx files" 
    if ext == '.csv':
        df = pd.read_csv(filename)
    else:
        df = pd.read_excel(filename)    
    return df