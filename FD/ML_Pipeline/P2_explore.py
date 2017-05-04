import matplotlib.pyplot as plt
import re

def basics(df):
    '''
    Given:
        df, a dataframe 

    Prints:
        keys of the df
        first five observations
        number of observations 
        descriptive statistics
    '''
    print('Observations:\n' + '{}\n'.format(df.shape[0]))
    print('{} features:'.format(df.shape[1]))
    i = 1
    for key in df.keys():
        print('    {}) {}'.format(i, key))
        i += 1
    print('\n')
    print('Sample observations:\n' + '{}\n'.format(df.head()))

def stats(df):
    '''
    Given:
        df, a dataframe 

    Prints:
        keys of the df
        first five observations
        number of observations 
        descriptive statistics
    '''
    summary = df.describe().T
    summary['median'] = df.median()
    summary['skew'] = df.skew() # skew of normal dist = 0
    summary['kurtosis'] = df.kurt() # kurtosis of normal dist = 0
    summary['missing_vals'] = df.count().max() - df.describe().T['count']
    print('Descriptive statistics:\n' + '{}\n'.format(summary.T))
    
def corr(df):
    print('Correlation matrix:\n' + '{}\n'.format(df.corr()))

def plot(df):
    '''
    Given:
        df, a pd.dataframe 

    Generates histograms in a separate folder
    '''
    print('Check the current folder for plotted graphs of these features.')
    for feature in df.keys():
        unique_vals = len(df[feature].value_counts())
        figure = plt.figure()
        if unique_vals == 1:
            df.groupby(feature).size().plot(kind='bar')
        elif unique_vals < 15:
            bins = unique_vals
            df[feature].hist(xlabelsize=10, ylabelsize=10, bins=unique_vals)
        else:
            df[feature].plot.hist()

        # Plot description labels
        clean_name = ' '.join(re.findall(r"[A-Za-z0-9][A-Za-z0-9][^A-Z0-9]*",
                                         feature))
        plt.ylabel('Frequency')
        plt.title(clean_name)
        plt.savefig('histograms/{}'.format(feature) + '_hist')
        plt.close()
   