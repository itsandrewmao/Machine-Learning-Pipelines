import pandas as pd

# 1. Read Data: For this assignment, assume input is CSV and write a function 
# that can read a csv into python

def read_data(filename):
    if '.csv' in filename:
        data = pd.read_csv(filename)
    elif '.dta':
        data = pd.read_stata(filename)

    return data