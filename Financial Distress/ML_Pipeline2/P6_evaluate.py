import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import (roc_curve,
                             precision_recall_curve, 
                             average_precision_score,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             auc, 
                             classification_report, 
                             confusion_matrix, 
                             f1_score)
# credits to https://github.com/yhat/DataGotham2013/blob/master/notebooks/8%20-%20Fitting%20and%20Evaluating%20Your%20Model.ipynb

def gen_precision_recall_plots(X, y, best_models):
    '''
    '''
    X_train, X_test, y_train, y_test = \
                        train_test_split(X, y, test_size=0.2, random_state=0)

    for name, d in best_models.items():
        clf = classifiers[name]
        p = eval(d['parameters'])
        clf.set_params(**p)
        y_true = y_test
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
        else:
            y_prob = clf.fit(X_train, y_train).decision_function(X_test)
        plot_precision_recall_n(y_true, y_prob, name)

def evaluate_classifier(y_test, yhat):
    '''
    For an index of a given classifier, evaluate it by various metrics
    '''
    # Metrics to evaluate
    metrics = {'precision': precision_score(y_test, yhat),
                'recall': recall_score(y_test, yhat),
                'f1': f1_score(y_test, yhat),
                'auc': roc_auc_score(y_test, yhat)}
    return metrics

def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    '''
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

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