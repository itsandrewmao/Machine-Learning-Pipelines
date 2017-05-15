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

def gen_precision_recall_plots(X, y, best_models):
    '''
    Plot precision recall graph given y_true, y_prob, and name of model
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

def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    Plot precision recall graph given y_true, y_prob, and name of model
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
