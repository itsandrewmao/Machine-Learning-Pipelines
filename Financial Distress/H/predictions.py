"""
Module for predictions.
"""

import time
import random
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import optimize

from sklearn.metrics import *
from sklearn.model_selection import ParameterGrid, train_test_split

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def clf_loop(models_to_run, clfs, grid, X_train, X_test, y_train, y_test, list_ks, list_thresholds, print_plots = False):
    """
    Loops through classifiers and stores metrics in pandas df.
    Df gets returned.
    Adjusted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py

    In:
        - models_to_run: (list) of models to run
        - clfs: (dict) of classifiers
        - grid: (dict) of classifiers with set of parameters to train on
        - X_train: features from training set
        - X_test: features from test set
        - y_train: targets of training set
        - y_test: targets of test set
        - list_ks: list of k's to use for precision at k calculations
        - list_thresholds: list of thresholds to use for binary decision (1 or not)
        - print_plots: (bool) whether or not to print plots
    Out:
        - pandas df
    """
    precision_at_k_lst = ["p_at_" + str(k) for k in list_ks]
    recall_at_k_lst = ["rec_at_" + str(k) for k in list_ks]
    results_df = pd.DataFrame(columns=['model_type', 'clf', 'parameters', 'train_time',
                                        'predict_time', 'threshold', 'auc-roc'] + precision_at_k_lst + recall_at_k_lst)

    for index, clf in enumerate([clfs[x] for x in models_to_run]):

        print(models_to_run[index])

        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            try:
                clf.set_params(**p)

                start_time_training = time.time()
                clf.fit(X_train, y_train)
                train_time = time.time() - start_time_training

                start_time_predicting = time.time()
                y_pred_probs = clf.predict_proba(X_test)[:,1]
                predict_time = time.time() - start_time_predicting

                roc_score = roc_auc_score(y_test, y_pred_probs)
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))

                for threshold in list_thresholds:

                    precision_at_k_scores = []
                    recall_at_k_scores = []

                    for first_k in list_ks:
                        precision, recall = calc_precision_recall(y_pred_probs_sorted, y_test_sorted, threshold, first_k)
                        precision_at_k_scores.append(precision)
                        recall_at_k_scores.append(recall)

                    results_df.loc[len(results_df)] = [models_to_run[index], clf, p,
                                                        train_time, predict_time, threshold,
                                                        roc_score] + precision_at_k_scores + recall_at_k_scores

                if print_plots:
                    _ = plot_precision_recall_n(y_test, y_pred_probs, clf)

            except IndexError as e:
                print('Error:', e)
                continue

    return results_df


def plot_precision_recall_n(y_true, y_prob, model_name):
    """
    Function to plot precision recall curve.
    Adjusted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py
    """
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_prob)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_prob)

    for value in pr_thresholds:
        num_above_thresh = len(y_prob[y_prob >= value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    _ = plt.clf()
    fig, ax1 = plt.subplots()
    _ = ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    _ = ax1.set_xlabel('percent of population')
    _ = ax1.set_ylabel('precision', color='b')

    ax2 = ax1.twinx()
    _ = ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    _ = ax2.set_ylabel('recall', color='r')
    _ = ax1.set_ylim([0,1])
    _ = ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    _ = ax2.set_xlim([0,1])
    _ = ax2.set_ylim([0,1])
    _ = ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    _ = plt.title(model_name)
    _ = plt.show()


def define_clfs_params(grid_size):
    """
    Defines classifiers and param grid that are used
    to find best model.
    Adjusted from: https://github.com/rayidghani/magicloops/blob/master/magicloops.py
    """

    clfs = {
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'DT': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'NN': MLPClassifier()
            }

    large_grid = {
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [None, 1,5,10,20,50,100], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,5,10]},
    'RF': {'n_estimators': [1,10,100,1000,10000], 'max_depth': [None, 1,5,10,20,50,100], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,5,10]},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'SVM': {'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10], 'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate': [0.001,0.01,0.05,0.1,0.5],'subsample': [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB': {},
    'KNN': {'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'NN': {'activation': ["tanh", "relu"], 'solver': ["lbfgs", "sgd"], "alpha": [0.0001, 0.1],
            'learning_rate': ["constant", "adaptive"], 'learning_rate_init': [0.001, 0.01, 0.1]}
    }

    small_grid = {
    'LR': { 'penalty': ['l1','l2'], 'C': [0.001,0.1,1,10]},
    'DT': {'criterion': ['gini'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,5,10]},
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,10]},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'SVM' :{'C' :[0.0001,0.001,0.01,0.1,1,10], 'kernel':['linear', 'poly']},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5], 'subsample': [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB': {},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100], 'weights': ['uniform','distance'], 'algorithm': ['auto','ball_tree','kd_tree']},
    'NN': {'activation': ["tanh", "relu"], 'solver': ["lbfgs", "sgd"], "alpha": [0.0001, 0.1],
            'learning_rate': ["constant", "adaptive"], 'learning_rate_init': [0.001, 0.01, 0.1]}
    }

    test_grid = {
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'], 'min_samples_split': [10]},
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'], 'min_samples_split': [10]},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'SVM' :{'C' :[0.01], 'kernel':['linear']},
    'GB': {'n_estimators': [1], 'learning_rate': [0.1], 'subsample': [0.5], 'max_depth': [1]},
    'NB': {},
    'KNN' :{'n_neighbors': [5], 'weights': ['uniform'], 'algorithm': ['auto']},
    'NN': {'activation': ["tanh", "relu"], 'solver': ["lbfgs", "sgd"], "alpha": [0.0001, 0.1],
            'learning_rate': ["constant", "adaptive"], 'learning_rate_init': [0.001, 0.01, 0.1]}
    }

    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0


class Zero_Predictor:
    """
    Class for predictor that always predicts 0.
    """
    def __init__(self):
        self.probability = 0

    def predict(self, X = None):
        """
        Predicts 0 once or for len of input.
        """
        if X is not None:
            predictions = [0 for i in range(len(X))]
            return np.asarray(predictions)
        else:
            return 0

    def fit(self, *args):
        pass

    def score(self, X, y, score='accuracy'):
        """
        Making predictions and calculating score on X.
        In:
            - X: features
            - y: true target
            - score: metric to use for evaluation.
                    options: accuracy, precision
        """
        predictions = self.predict(X)
        true_value = np.asarray(y)

        if score == "accuracy":
            return 1 - sum(np.absolute(predictions - true_value)) / len(predictions)

        elif score == "precision":
            return 0


class Average_Predictor:
    """
    Class for predictor that makes predictions according
    to average of target vector: how well would we do if
    we randomly predict according to mean of training data?
    """
    def __init__(self):
        self.probability = 0

    def fit(self, target):
        """
        Function to 'train' the model, i.e. determine the
        mean of the target variable.
        """
        self.probability = target.mean()

    def predict(self, X = None):
        """
        Function to make predictions by creating random
        number and seeing if generated number is greater than
        mean of target variable.
        """
        if X is not None:
            predictions = [1 if random.random() < self.probability else 0 for i in range(len(X))]
            return np.asarray(predictions)
        else:
            if random.random() < self.probability:
                return 1
            else:
                return 0

    def score(self, X, y, score='accuracy'):
        """
        Making predictions and calculating score on X.
        In:
            - X: features
            - y: true target
            - score: metric to use for evaluation.
                    options: accuracy, precision
        """
        predictions = self.predict(X)
        true_value = np.asarray(y)

        if score == "accuracy":
            score = 1 - sum(np.absolute(predictions - true_value)) / len(predictions)

        elif score == "precision":
            tp = fp = 0
            for i in range(len(predictions)):
                if predictions[i] == 1 and true_value[i] == 1:
                    tp += 1
                elif predictions[i] == 1:
                    fp += 1
            if tp + fp == 0:
                score = 0
            else:
                score = tp / (tp + fp)

        return score


def report(results, n_top=3):
    """
    Source:
    http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#sphx-glr-auto-examples-model-selection-randomized-search-py
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def calc_precision_recall(predicted_values, y, threshold, top_k):
    """
    Calculates precision and recall for given threshold.
    In:
        - predicted_values: numpy array of predicted scores
        - y: target values
        - threshold: threshold to use for calculation
        - top_k: int or "All" - how many first entries are we considering?
    Out:
        - (precision_score, recall_score)
    """
    y = np.asarray(y)
    x = [1 if predicted_value >= threshold else 0 for predicted_value in predicted_values]

    tp = fp = fn = 0

    if top_k == "All" or top_k == None:
        top_k = len(x)

    for i in range(top_k):
        if x[i] == 1 and y[i] == 1:
            tp += 1
        elif x[i] == 1:
            fp += 1
        elif x[i] == 0 and y[i] == 1:
            fn += 1
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    return precision, recall


def plot_precision_recall_threshold(threshold_list, precision_list, recall_list):
    """
    Function to plot precision recall by threshold curve.
    In:
        - threshold_list: list of threshold values
        - precision_list: list of achieved precision
        - recall_list: list of achieved recall
    Out:
        - plot
    """

    fig, ax1 = plt.subplots()

    _ = ax1.plot(threshold_list, precision_list, 'b.')
    _ = ax1.set_xlabel('threshold')

    _ = ax1.set_ylabel('precision', color='b')
    _ = ax1.set_ylim(0,1.1)
    _ = ax1.set_yticks([0,0.2,0.4,0.6,0.8,1])
    _ = ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    _ = ax2.plot(threshold_list, recall_list, 'r.')

    _ = ax2.set_ylabel('recall', color='r')
    _ = ax2.set_ylim(0,1.1)
    _ = ax2.set_yticks([0,0.2,0.4,0.6,0.8,1])
    _ = ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()


def precision_top_x(predictions, y, x):
    """
    Calculates precision for x first entries.
    In:
        - predictions: array of predicted scores
        - y: target values
        - x: number of first entries to consider
    Out:
        - precision_score
    """
    y = np.asarray(y)
    predictions = np.asarray(predictions)

    tp = fp = 0
    for i in range(x):
        if predictions[i] == 1 and y[i] == 1:
            tp += 1
        elif predictions[i] == 1:
            fp += 1
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    return precision
