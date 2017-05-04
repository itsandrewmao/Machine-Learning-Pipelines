import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, 
                              GradientBoostingClassifier,
                              BaggingClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (roc_auc_score,
                             roc_curve,
                             precision_recall_curve, 
                             average_precision_score,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             auc, 
                             classification_report, 
                             confusion_matrix, 
                             f1_score)

# Classifiers to test
classifiers = {'LR': LogisticRegression(),
               'KNN': KNeighborsClassifier(),
               'DT': DecisionTreeClassifier(),
               'SVM': SVC(),
               'RF': RandomForestClassifier(),
               'GB': GradientBoostingClassifier()}

grid = {'LR': {'penalty': ['l1', 'l2'], 
               'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]}, 
        'KNN': {'n_neighbors': [1, 5, 10, 25, 50, 100], 
                'weights': ['uniform', 'distance'], 
                'algorithm': ['auto', 'ball_tree', 'kd_tree']},
        'DT': {'criterion': ['gini', 'entropy'], 
               'max_depth': [1, 5, 10, 20, 50, 100], 
               'max_features': ['sqrt', 'log2'], 
               'min_samples_split': [2, 5, 10]},
        'SVM' : {'C' : [0.1, 1]},
        'RF': {'n_estimators': [1, 10], 
               'max_depth': [1, 5, 10], 
               'max_features': ['sqrt', 'log2'], 
               'min_samples_split': [2, 5, 10]},
        'GB': {'n_estimators': [1, 10], 
               'learning_rate' : [0.1, 0.5], 
               'subsample' : [0.5, 1.0], 
               'max_depth': [1, 3, 5]}
        }


def classify(X, y, models, iters, threshold, metrics):
    '''
    Given a dataframe of features (X), a dataframe of the label (y), and a list 
    of models to run, an integer indicating iterations, a threshold

    Returns:
        A dictionary that contains each classifier's performace on the given
        evaluation metrics, the best models, and the winning classifier.
    '''
    
    rv = {}
    
    # First, we construct the train and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.3, 
                                                        random_state = 5678)
        
    # For every classifier, we try all paramter combinations possible
    for index, clf in enumerate([classifiers[x] for x in models]):
        name = models[index]
        print("{} is running...".format(name))
        parameter_values = grid[name]
        rv[name] = {}
        
        # We initiate a nested loop to run our model with all specified parameters
        for p in ParameterGrid(parameter_values):
            precision_per_iter = []
            recall_per_iter = []
            f1_per_iter = []
            auc_per_iter = []
            time_per_iter = []
            avg_metrics = {}
            rv[name][str(p)] = {}
            results = rv[name][str(p)]
            clf.set_params(**p)

            # Now, we run i number of iterations
            for run in range(iters): 
                try:
                    start_time = time.time()
                    
                    # get the predicted results from the model
                    if hasattr(clf, 'predict_proba'):
                        yscores = clf.fit(X_train, 
                                          y_train).predict_proba(X_test)[:,1]
                    else:
                        yscores = clf.fit(X_train,
                                          y_train).decision_function(X_test)
                    yhat = np.asarray([1 if run >= threshold else 0 for run in yscores])
                    end_time = time.time()

                    # Finally, we can take down metrics
                    metrics = evaluate_classifier(y_test, yhat)
                    for m, value in metrics.items():
                        eval('{}_per_iter'.format(m)).append(value)
                    time_per_iter.append(end_time - start_time)
                    
                except IndexError:
                    print('Error')
                    continue
            
            avg_metrics['time'] = np.mean(time_per_iter)
            results['time'] = np.mean(time_per_iter)

            # We store average metrics of model p
            for m in metrics:
                avg_metrics[m] = np.mean(eval('{}_per_iter'.format(m)))
                results[m] = avg_metrics[m]

        print('Finished running {}!'.format(name))

    # Last but not least, we can save our results as a json to view later
    with open('Model_Comparison.json', 'w') as fp:
        json.dump(rv, fp)

    return rv

def select_best_models(results, models, d_metric):
    columns = ['auc', 'f1', 'precision', 'recall', 'time', 'parameters']
    table = pd.DataFrame(index = models, columns = columns)
    best_metric = 0
    best_models = {}
    rv = {}

    for model, iters in results.items():
        top_intra_metric = 0
        best_models[model] = {}
        for params, metrics in iters.items():
            header = [key for key in metrics.keys()]
            if metrics[d_metric] > top_intra_metric:
                top_intra_metric = metrics[d_metric]
                best_models[model]['parameters'] = params
                best_models[model]['metrics'] = metrics

        to_append = [value for value in best_models[model]['metrics'].values()]
        to_append.append(best_models[model]['parameters'])
        table.loc[model] = to_append
        if top_intra_metric > best_metric:
            best_metric = top_intra_metric
            best_model = model, params
    
    rv["table"] = table
    rv["best_models"] = best_models
    rv["best"] = (best_model, best_metric)
        
    return rv

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