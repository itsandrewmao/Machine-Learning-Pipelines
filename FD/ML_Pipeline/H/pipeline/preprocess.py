### Machine Learning for Public Policy
### Pipeline: Pre-process data
### Héctor Salvador López

import numpy as np

def impute_csv(data, fill_type='mean', val=0):
	'''
	Takes:
		data, a pd.dataframe
		fill_type, a string with the type of imputation to be performed
		val, an optional value when the fill_type is set to 'value'
	'''
	fill_types = ['mean', 'median', 'mode', 'value']
	assert fill_type in fill_types

	j = 0

	# for every feature with null values, replace the null values with the fill_type
	for feature in data.keys():
		has_nulls = data[feature].isnull().values.any()
		print('{}) {} has null values: {}.'.format(j, feature, has_nulls))
		if has_nulls:
			print('  Filling nulls with {}.'.format(fill_type))
			if fill_type == 'mean':
				filling = data[feature].mean()
			elif fill_type == 'median':
				filling = data[feature].median()
			elif fill_type == 'mode':
				filling = data[feature].mode()
			elif fill_type == 'value':
				filling = val
			data[feature] = data[feature].fillna(filling)
		j += 1


def transform_feature(data, feature, f):
	'''
	Takes:
		data, a pd.dataframe
		feature, the feature we want to transform
		f, any real function (e.g. lambda x: x**2 + 1)
	'''
	f_feat = 'f({})'.format(feature) #put a name on the function
	data[f_feat] = data[feature].apply(lambda x: f(x))
