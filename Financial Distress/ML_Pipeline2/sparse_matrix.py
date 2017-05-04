'''
A container class for CSC sparse matrices. Note that sparse matrices can only
hold floats or integers; string and datetime data will need to be cleaned.
Currently, the class accounts for a datetime hours column.

Constructor is called with an exclude_before (int), which tells
the class to move columns before exclude_before into the 'excluded' attribute.
This is currently set to 2, such that the constructor moves cell_id and
date to the excluded columns.

Attributes:
    - data: a csc matrix
    - columns: the column names, in order
'''
import csv
import numpy as np
import datetime
from scipy import sparse

class SparseMatrix:

    def __init__(self, exclude=2):
        '''
        Constructor.
        '''
        self.data = None
        self.columns = None
        self.excluded = None
        self.excluded_columns = None
        self.label = None
        self.exclude_before = exclude

    def get(self, column):
        '''
        Returns a column from the data by column name as a (m x 1) matrix.
        '''
        if self.data == None:
            print 'No data in matrix.'
        if column not in self.columns:
            print 'Column does not exist.'
        else:
            i = self.columns.index(column)
            return self.data.getcol(i)

    def get_all_except(self, column):
        '''
        Returns all columns except the one specified.
        '''
        if self.data == None:
            print 'No data in matrix.'
        if column not in self.columns:
            print 'Column does not exist.'
        else:
            i = self.columns.index(column)
            halfone = self.data[:,:i]
            halftwo = self.data[:,i+1:]
            return sparse.hstack([halfone, halftwo])

    def load_csv(self, filename, headers = True):
        '''
        Loads data from csv.
        '''
        rv = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            if headers:
                self.columns = reader.next()
            for row in reader:
                rv.append(row)
        self.load_tuples(rv)

    def load_tuples(self, tuples):
        '''
        Takes a list of tuples and turns into a sparse matrix.
        '''
        array = np.array(tuples)[:,self.exclude_before:]
        self.excluded = np.array(tuples)[:,:self.exclude_before]
        array_columns = self.columns[self.exclude_before:]
        self.excluded_columns = self.columns[:self.exclude_before]
        if 'hr' in self.columns:
            i = array_columns.index('hr')
            array[:,i] = [datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").hour for t in array[:,i]]
        array[array==''] = 0 # Convert all missing values to 0
        self.data = sparse.csc_matrix(array.astype('float'))
        self.columns = array_columns
