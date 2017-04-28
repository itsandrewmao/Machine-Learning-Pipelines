import pandas as pd
import matplotlib.pyplot as plt

#Explore independent variables
def create_var_dict(df, i):
    '''
    ***Dependent variable y (binary) takes the i-th
    column in the DataFrame***
    Prints out the names of independent variables.
    Returns corresponding dictionary.
    '''
    d = {}
    d['y'] = df.columns[i-1]
    print('y: '+d['y'])
    x = 1
    for k in range(i, len(df.columns)):
        var = 'x'+str(x)
        d[var]=df.columns[k]
        x += 1
        print(var+": "+d[var])
    return d


#Graph 1:
#Figure out the difference in mean of x's of interest
#between y=0 and y=1
def diff_in_mean(df, var_lst, d):
    '''
    var_lst: list of integers corresponding to
        independent variables of interest.
        E.g., interested in x3,x5,x8
        then, var_lst = [3,5,8]
    d: the independent variable dictionary from
        create_var_dict function.
    '''
    col_names = [d['y']]
    for i in var_lst:
        col_names.append(d['x'+str(i)]) 
    print(df[col_names].groupby(d['y']).mean())


#Graph 2:
#Show the distribution of x's of interest
#using a boxplot.
def x_dist(df, var_lst, d):
    '''
    Plots box charts.
    
    *Refer to diff_in_mean function
    for requirements of var_lst and d.
    '''
    for i in var_lst:
        plt.figure()
        name = d['x'+str(i)]
        df[name].plot(kind='box')
        plt.title('Boxplot of '+ name)


#Graph 3:
#For an x's of interest
#show the percentage of y=0 vs. y=1
#for a certain value of x.
def comparison_all_values(df, var_lst, d):
    '''
    Prints bar charts.
    
    *Refer to diff_in_mean function
    for requirements of var_lst and d.
    '''
    for i in var_lst:
        plt.figure()
        name = d['x'+str(i)]
        ct = pd.crosstab(df[name],df[d['y']].astype(bool))
        ct.div(ct.sum(1).astype(float), axis=0).plot(
            kind='bar', figsize=(20,5), stacked=True)
