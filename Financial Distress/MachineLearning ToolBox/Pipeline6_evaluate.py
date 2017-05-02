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

    # We should also look at Acscuracy
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
    
def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    preds_at_k = generate_binary_at_k(y_scores, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true, preds_at_k)
    return precision