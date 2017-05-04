### Machine Learning for Public Policy
### Pipeline: Generate features
### Héctor Salvador López

### Functions for dataframes
import pandas as pd

def binning(data, feature, type_cut, quantiles = 0.5, bins = 1):
	'''
	Takes:
		data, a pd.dataframe 
		feature, a string with the name of the feature to put into bins
		type_cut, quantiles or bins
		quantiles, an int or array of quantiles
		bins, an int or sequence of scalars
	'''
	valid_cuts = ['quantiles', 'bins']
	assert type_cut in valid_cuts

	bins = 'bins_{}'.format(feature)
	if type_cut == 'quantiles':
		data[bins] = pd.qcut(data[feature], quantiles, labels=False)
	elif type_cut == 'n':
		data[bins] = pd.cut(data[feature], bins, labels=False)


def binarize(data, feature, control):
	'''
	Takes:
		data, a pd.dataframe 
		feature, a string of the name of the feature to binarize
		control, the value of a feature that will have a zero value in the
			new binarized feature 
	'''
	data[feature] = data[feature].apply(lambda x: 0 if x == control else 1)
