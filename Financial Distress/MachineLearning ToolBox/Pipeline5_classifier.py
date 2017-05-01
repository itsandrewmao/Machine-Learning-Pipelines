from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
#Bag,Boost,RF
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def all_data(df, var_lst, d):
    '''
    Determine which independent variables
    to be used in the logit regression.

    Return dataframes of all available data
    for y and x's.
    '''
    Y = df[d['y']]
    ind_vars = []
    for i in var_lst:
        ind_vars.append(d['x'+str(i)])
    X = df[ind_vars]
    return X, Y

def classify(df, features, label, method, bagging=False):
    '''
    Given training and testing data for independent variables (features),
    training data for dependent variable, and classifying method,
    return model, X_test, y_test
    '''
    if method == "KNN":
        model = KNeighborsClassifier(n_neighbors = 13, 
                                     metric = 'minkowski', 
                                     weights = 'distance')
    elif method == "Tree":
        if bagging:
            model = BaggingClassifier(model, 
                                      n_estimators = 10, 
                                      max_samples=0.65, 
                                      max_features=1.)
        else:
            model = tree.DecisionTreeClassifier(max_depth = 20,
                                            class_weight = 'balanced')
        
    elif method == "GB":
        model = GradientBoostingClassifier()
    elif method == "Logit":
        if bagging:
            model = BaggingClassifier(model, 
                                      n_estimators = 10, 
                                      max_samples=0.65, 
                                      max_features=1.)
        else:
            model = LogisticRegression('l2',
                                       C = 1,
                                       class_weight = 'balanced')
    elif method == "RF":
        model = RandomForestClassifier(n_estimators=2, 
                                     max_features = 'sqrt', 
                                     max_depth = 20, 
                                     min_samples_split = 2 ,
                                     class_weight = 'balanced')
        
    else:
        raise ValueError('{} not currently avaliable'.format(method))
        
    y = df[label]
    X = df[features]
    
    X_train, X_test, y_train, y_test = train_test_split(y, 
                                                        X, 
                                                        test_size = 0.20,
                                                        random_state = 311)
    ''' old method
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    '''
    
    model.fit(X_train, y_train)

    return model, X_test, y_test

''' Old code

def bag(df, features, label, method):
    '''
    '''
    
    if method == "Tree":
        model = DecisionTreeClassifier(max_depth = 20, 
                                       class_weight = 'balanced')
    elif method == "Logit":
        model = LogisticRegression('l2', 
                                C = 1, 
                                class_weight = 'balanced')
        
        bagging = BaggingClassifier(lr, 
                                    n_estimators = 10, 
                                    max_samples=0.65, 
                                    max_features=1.)
    else:
        raise ValueError('{} not currently avaliable'.format(method))
    

    bagging.fit(X_train, y_train)
    score = bagging.score(X_test, y_test)
    
        
    Y = df[label].head()
    X = df[features].head()
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size=0.2, 
                                                        random_state=2)

def boost(df, features, label, method):
    '''
    '''
    AdaBoost = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1,
                                                         class_weight = 'balanced'),
                                  n_estimators=5,
                                  learning_rate=1)
    
    AdaBoost.fit(X_train, y_train)
    score = AdaBoost.score(X_test, y_test)
    
    return score
    
def rf(df, features, label, method):
    '''
    '''
    
    rforest = RandomForestClassifier(n_estimators=2, 
                                     max_features = 'sqrt', 
                                     max_depth = 20, 
                                     min_samples_split = 2 ,
                                     class_weight = 'balanced')

    rforest.fit(X_train, y_train)
    score = rforest.score(X_test, y_test)
    
    return score
'''