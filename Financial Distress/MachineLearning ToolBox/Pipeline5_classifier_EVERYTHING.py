import time
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

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

def display_importance(df, label, features, method):
    '''
    Given dataframe, label, and list of features,
    plot a graph to rank variable importance
    '''
    
    if method == "Decision Tree":
        model = DecisionTreeClassifier()
    elif method == "Gradient Boosting":
        model = GradientBoostingClassifier()
    elif method == "Random Forest":
        model = RandomForestClassifier()
    else:
        raise ValueError('{} not currently avaliable'.format(method))
        
    model.fit(df[features], df[label])
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    padding = np.arange(len(features)) + 0.5
    plt.barh(padding, importances[sorted_idx], align='center')
    plt.yticks(padding, np.asarray(features)[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance for {}".format(method))


def make_model(method):

    if method == 'RF':
        model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    elif method == 'BA':
        model = BaggingClassifier(DecisionTreeClassifier(max_depth=1), n_estimators = 5, max_samples=0.65, max_features=1.)
    elif method == 'AB':
        model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
    elif method == 'LR':
        model = LogisticRegression(penalty='l1', C=1e5)
    elif method == 'SVM':
        model = SVC(kernel='linear', probability=True, random_state=0)
    elif method == 'DT':
        model = DecisionTreeClassifier()
    elif method == 'KNN':
        model = KNeighborsClassifier(n_neighbors=3)

    return model



def classify(df, features, label, method_dict):
    '''
    Given training and testing data for independent variables (features),
    training data for dependent variable, and classifying method,
    return model, X_test, y_test
    
    if m[method] == "KNN":
        model = KNeighborsClassifier(n_neighbors = 13, 
                                     metric = 'minkowski', 
                                     weights = 'distance')
    elif m[method] == "Tree":
        if bagging:
            model = BaggingClassifier(model, 
                                      n_estimators = 10, 
                                      max_samples=0.65, 
                                      max_features=1.)
        else:
            model = tree.DecisionTreeClassifier(max_depth = 20,
                                            class_weight = 'balanced')
        
    elif m[method] == "GB":
        model = GradientBoostingClassifier()
    elif m[method] == "Logit":
        if bagging:
            model = BaggingClassifier(model, 
                                      n_estimators = 10, 
                                      max_samples=0.65, 
                                      max_features=1.)
    else:
            model = LogisticRegression('l2',
                                       C = m[C],
                                       class_weight = m[class_weight)
    elif m[method] == "RF":
        model = RandomForestClassifier(n_estimators=2, 
                                     max_features = 'sqrt', 
                                     max_depth = 20, 
                                     min_samples_split = 2 ,
                                     class_weight = 'balanced')
    '''
    m = method_dict
    
    if m[method] == "KNN":
        model = KNeighborsClassifier(n_neighbors = m[n_neighbors], 
                                     metric = m[metric], 
                                     weights = m[weights])
        if scale:
            min_max_scaler = preprocessing.MinMaxScaler()
            features = min_max_scaler.fit_transform(features)
            
    elif m[method] == 'SVM':
        model = SVC(kernel = m[kernel],
                    probability = m[probability], 
                    random_state = m[random_state])
        
        if scale:
            min_max_scaler = preprocessing.MinMaxScaler()
            features = min_max_scaler.fit_transform(features)
        
    elif m[method] == "Tree":
        model = tree.DecisionTreeClassifier(max_depth = m[max_depth],
                                            class_weight = m[class_weight])
        if m[bagging]:
            model = BaggingClassifier(model, 
                                      n_estimators = m[n_estimators], 
                                      max_samples = m[max_samples], 
                                      max_features = m[max_features])
    
    elif m[method] == "Logit":
        model = LogisticRegression('l2',
                                   C = m[C],
                                   class_weight = m[class_weight])

        if m[bagging]:
            model = BaggingClassifier(model, 
                                      n_estimators = m[n_estimators], 
                                      max_samples = m[max_samples], 
                                      max_features = m[max_features])
    
    elif m[method] == "RF":
        model = RandomForestClassifier(n_estimators = m[n_estimators], 
                                       max_features = m[max_features], 
                                       max_depth = m[max_depth], 
                                       min_samples_split = m[min_samples_split],
                                       class_weight = m[class_weight])
        
    else:
        raise ValueError('{} not currently avaliable'.format(method))
        
    ''' old method
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    probs = model.predict_proba(X_test)
    '''
    
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

''' Old code

def bag(df, features, label, method):
    
  
    
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
    
    AdaBoost = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1,
                                                         class_weight = 'balanced'),
                                  n_estimators=5,
                                  learning_rate=1)
    
    AdaBoost.fit(X_train, y_train)
    score = AdaBoost.score(X_test, y_test)
    
    return score
    
def rf(df, features, label, method):
    
    
    rforest = RandomForestClassifier(n_estimators=2, 
                                     max_features = 'sqrt', 
                                     max_depth = 20, 
                                     min_samples_split = 2 ,
                                     class_weight = 'balanced')

    rforest.fit(X_train, y_train)
    score = rforest.score(X_test, y_test)
    
    return score
'''