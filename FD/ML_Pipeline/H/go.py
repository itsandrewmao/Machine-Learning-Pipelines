### Machine Learning for Public Policy
### Homework 3
### Héctor Salvador López
### Code for the pipeline was significantly inspired on:
### 	/rayidghani/magicloops/blob/master/magicloops.py
### 	/BridgitD/Machine-Learning-Pipeline/blob/master/pipeline.py
### 	/danilito19/CAPP-ML-dla/blob/master/pa3/workflow.py
###		/ladyson/ml-for-public-policy/blob/master/PA3/pipeline.py
### 	/demunger/CAPP30254/blob/master/HW3/hw3.py
### 	/aldengolab/ML-basics/blob/master/pipeline/model.py

import math
import numpy as np
import pandas as pd
from pipeline import reading, explore, preprocess, features, classify
from sklearn.cross_validation import train_test_split

fts = ['RevolvingUtilizationOfUnsecuredLines', 
        'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 
        'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 
        'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 
        'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

label = 'SeriousDlqin2yrs'
filename = 'pipeline/data/cs-training.csv'
models = ['LR', 'KNN', 'DT', 'SVM', 'RF', 'GB']
metrics = ['precision', 'recall', 'f1', 'auc']

def go(filename, features, label, models, metric):
	# read dataset
	df = reading.read(filename)

	# divide dataset to train and test
	xtrain, xtest, ytrain, ytest = train_test_split(df[features], df[label])
	train = pd.concat([ytrain, xtrain], axis = 1)
	test = pd.concat([ytest, xtest], axis = 1)
	df = train

	# details on exploration can be found on the jupyter notebook

	# impute null values with mean value and transform income to log(income)
	preprocess.impute_csv(df)
	preprocess.transform_feature(df, 'MonthlyIncome', lambda x: math.log(x + 1))
	fts.append(df.keys()[-1])

	# create a feature of income quartile
	features.binning(df, 'f(MonthlyIncome)', 'quantiles', [0, 0.25, 0.5, 0.75, 1])
	fts.append(df.keys()[-1])

	# deploy classifiers
	all_models = classify.classify(df[features], df[label], models, 3, 0.05, metrics)
	table, best_models, winner = \
		classify.select_best_models(all_models, models, metric)

	# get plots 
	classify.gen_precision_recall_plots(df[features], df[label], best_models)

	return all_models, table, best_models, winner

