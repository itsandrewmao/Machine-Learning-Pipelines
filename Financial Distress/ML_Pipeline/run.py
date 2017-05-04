'''
Runs the model loop for CivicScape.
'''

import csv
import argparse
import numpy as np
from model_loop import ModelLoop
from sparse_matrix import SparseMatrix

def pipeline(args):
    '''
    Runs the model loop.

    If you wish to edit any of the parameters for the models, please edit the
    model_loop.py file directly.
    '''
    train_data = SparseMatrix()
    train_data.load_csv(args.train_filename)
    y_train = train_data.get(args.label).todense()
    X_train = train_data.get_all_except(args.label)
    y_train[y_train>1] = 1 # Remove multiclass
    y_train = np.array(np.reshape(y_train, y_train.shape[0]))[0] # Correct shape

    test_data = SparseMatrix()
    test_data.load_csv(args.test_filename)
    y_test = test_data.get(args.label).todense()
    X_test = test_data.get_all_except(args.label)
    y_test[y_test>1] = 1 # Remove multiclass
    y_test = np.array(np.reshape(y_test, y_test.shape[0]))[0] # Correct shape

    loop = ModelLoop(X_train, X_test, y_train, y_test, args.models,
                     args.iterations, args.run_name,
                     args.thresholds, args.label, float(args.comparison),
                     args.project_folder)
    loop.run()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run a model loop.')
    parser.add_argument('train_filename', type=str,
                    help='Location of training data csv file')
    parser.add_argument('test_filename', type=str,
                    help='Location of test data csv file')
    parser.add_argument('--label', type=str,
                    help='Label for outcome column', default = 'label')
    parser.add_argument('--run_name', type=str,
                    help='Name of this run')
    parser.add_argument('--iterations', type=int,
                    help='Number of iterations', default = 50)
    parser.add_argument('--models', nargs='+',
                    help='Models to run', default = ['LR', 'RF', 'DT', 'SGD', 'SVM', 'AB', 'NN'])
    parser.add_argument('--thresholds', nargs='+', type=float,
                    help='Thresholds', default = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    parser.add_argument('--project_folder', type=str, default = './',
                    help='Relative project folder for results')

    args = parser.parse_args()
    print "\n========= NEW MODEL LOOP RUN {} =========\n".format(args.run_name.upper())
    for key in sorted(vars(args).keys()):
        print "{}: {}".format(key, vars(args)[key])
    pipeline(args)
