import matplotlib.pyplot as plt
import re

def tabular(df, varname):
    '''
    '''
    #number of complaints by type
    vcounts = df[varname].value_counts().to_frame()
    total = sum(vcounts[varname])
    vcounts['Percent'] = vcounts[varname]/total
    return vcounts.sort_values('Percent', ascending=False)

def histogram(df, varname):
    '''
    '''
    plt.clf()
    n, bins, patches = plt.hist(df[varname])
    clean_name = ' '.join(re.findall(r"[A-Za-z0-9][A-Za-z0-9][^A-Z0-9]*", varname))
    
    plt.title('Histogram for {}'.format(clean_name))
    plt.show()

def explore_varname(df,varname,label,method):
        
    assert method in ['line','bar','barh','box','kde','density','area','pie','scatter','hexbin'], "Graph Method not permitted. Try one of the following: 'line','bar','barh','box','kde','density','area','pie','scatter','hexbin'."
    
    assert varname in df.columns, "Column '{}' not in DataFrame".format(varname)
    
    rv = {}
    cols = [varname, label]
    m = df[cols].groupby(varname).mean()
    clean_name = ' '.join(re.findall(r"[A-Za-z0-9][A-Za-z0-9][^A-Z0-9]*", varname))
    
    rv["mean_dist"] = m
    rv["graph"] = m.plot(kind=method,
                         use_index=False,
                         figsize=(8,4),
                        title=clean_name)
    
    return rv