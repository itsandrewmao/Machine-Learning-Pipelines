# 6. Evaluate Classifier: you can use any metric you choose for this assignment 
# (accuracy is the easiest one). Feel free to evaluate it on the same data you 
# built the model on (this is not a good idea in general but for this assignment, 
# it is fine). We haven't covered models and evaluation yet, so don't worry about 
# creating validation sets or cross-validation. 

import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

# credits to https://github.com/yhat/DataGotham2013/blob/master/notebooks/8%20-%20Fitting%20and%20Evaluating%20Your%20Model.ipynb

def evaluate(model, X_te, y_te):
    '''
    Given the model and independent and dependent testing data,
    print out statements that evaluate classifier
    '''
    probs = model.predict_proba(X_te)
    
    plt.hist(probs[:,1])
    plt.xlabel('Likelihood of Significant Financial')
    plt.ylabel('Frequency')

    # We should also look at Accuracy
    print("Accuracy = " + str(model.score(X_te, y_te)))

    # Finally -- Precision & Recall
    y_hat = model.predict(X_te)
    print(classification_report(y_te, y_hat, labels=[0, 1]))
    
    y_hat = model.predict(X_te)    
    confusion_matrix = pd.crosstab(y_hat, 
                                   y_te, 
                                   rownames=["Actual"], 
                                   colnames=["Predicted"])
    print(confusion_matrix)

def plot_roc(probs, y_te):
    '''
    Plots ROC curve.
    '''
    plt.figure()
    fpr, tpr, thresholds = roc_curve(y_te, probs)
    roc_auc = auc(fpr, tpr)
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title("ROC Curve")
    pl.legend(loc="lower right")
    pl.show()