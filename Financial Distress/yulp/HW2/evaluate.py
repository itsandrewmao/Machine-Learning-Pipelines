import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

#Reference: https://github.com/yhat/DataGotham2013/blob/master/notebooks/8%20-%20Fitting%20and%20Evaluating%20Your%20Model.ipynb

def summary(model, X_test, Y_test):
    '''
    Evalute the logit model using testing data.
    '''
    #Plot histgram of the frequency of probability of y=1.
    probs = model.predict_proba(X_test)
    pl.hist(probs[:,1])
    plt.xlabel('Probability of Experiencing Financial Distress')
    plt.ylabel('Frequency')

    #Accuracy
    print("Accuracy = "+str(model.score(X_test, Y_test)))

    #Precision, Recall
    print(classification_report(Y_test, model.predict(X_test), labels=[False, True]))

def plot_roc(probs, Y_test):
    '''
    Plots ROC curve.

    *Credit goes to yhat.
    '''
    plt.figure()
    fpr, tpr, thresholds = roc_curve(Y_test, probs)
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