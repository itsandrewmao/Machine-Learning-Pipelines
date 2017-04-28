from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

def classify(df, features, label, cl_method):
    '''
    Given training and testing data for independent variables (features),
    training data for dependent variable, and classifying method,
    return model, X_test, y_test
    '''
    if cl_method == "KNN":
        model = KNeighborsClassifier(n_neighbors=13, 
                                     metric='minkowski', 
                                     weights='distance')
    elif cl_method == "Tree":
        model = tree.DecisionTreeClassifier()
    elif cl_method == "GB":
        model = GradientBoostingClassifier()
    elif cl_method == "Logit":
        model = LogisticRegression()
    else:
        raise ValueError('{} not currently avaliable'.format(cl_method))
        
    Y = df[label].head()
    X = df[features].head()
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size=0.2, 
                                                        random_state=2)
    
    model = model.fit(X_train, y_train)

    return model, X_test, y_test