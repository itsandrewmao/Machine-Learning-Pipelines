### Machine Learning for Public Policy
### Pipeline: Evaluate classifier
### Héctor Salvador López

from sklearn.metrics import accuracy_score 


def accuracy(observed, predicted):
	'''
	Takes:
		predicted, a list with floats or integers with predicted values
		observed, a list with floats or integers with observed values 

	Calculates the accuracy of the predicted values.
	'''
	return accuracy_score(observed, predicted)

def mse(observed, predicted):
	'''
	Takes:
		predicted, a list with floats or integers with predicted values
		observed, a list with floats or integers with observed values 

	Calculates the mean squared error of the predicted values.
	'''
	pass

def mad(predicted, observed):
	'''
	Takes:
		predicted, a list with floats or integers with predicted values
		observed, a list with floats or integers with observed values 

	Calculates the mean absolute deviation of the predicted values.
	'''
	pass
