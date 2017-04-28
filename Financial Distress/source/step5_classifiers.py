from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 5. Build Classifier: For this assignment, select any classifier you feel 
# comfortable with (Logistic Regression for example)

class MyClassifier(object):
    def __init__(self, X, Y, method, test_size=0.2, seed=None):
        self.X = X
        self.Y = Y
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=seed)
        
        if method == 'logistic':
            self.model = LogisticRegression()
            self.model.fit(self.X_train, self.Y_train)