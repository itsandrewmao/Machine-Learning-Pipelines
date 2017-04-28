# 2. Explore Data: You can use the code you wrote for assignment 1 here to 
# generate distributions and data summaries.

def describe_column(df, columnname):
    print(df[columnname].describe())
    print('')
    print('Unique values: {}'.format(len(df[columnname].value_counts())))
    print('')
    print('Missing values: {}'.format(sum(df[columnname].isnull())))
    
    #add number of unique values, missing values, other interesting stats

def tabular(df, columnname):
    #number of complaints by type
    ctcounts = df[columnname].value_counts().to_frame()
    total = sum(ctcounts[columnname])
    ctcounts['Percent'] = ctcounts[columnname]/total
    return ctcounts.sort_values('Percent', ascending=False)

def histogram(df, columnname):
    pass
    #create histogram
