import pandas as pd
from sklearn.cross_validation import train_test_split

from read import read_file
from describe import create_var_dict, diff_in_mean, x_dist, comparison_all_values
from process import check_missing_value, fill_in
from create_dummy import discretize, create_dummy
from logit import all_data, logit_regression
from evaluate import summary, plot_roc

get_ipython().magic('matplotlib inline')

#Step 1: Read Data
df = read_file("credit-data.csv")


#Step 2: Explore Data
d = create_var_dict(df, 2)  #independent variable is in the 2nd column
diff_in_mean(df, [2,6,11], d)
diff_in_mean(df, [4,10,8], d)
diff_in_mean(df, [1,5,7,9], d)
x_dist(df, [1,2,5,6,7,9,11],d)
comparison_all_values(df, [2,7,9,11,4,10,8],d)


#Step 3: Fill missing value
check_missing_value(df)
fill_in(df, 6, d, 'mean')
fill_in(df, 11, d, 'mean')


#Step 4: Create dummy variables
headers = discretize(df, 2, d, 4)
create_dummy(df,2,d,headers)


#Step 5: Classifier (Logit model)
X, Y = all_data(df, [1,4,5,6,7,8,9,10,11,12,13,14,15], d)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
model = logit_regression(X_train, Y_train)


#Step 6:
summary(model, X_test, Y_test)
plot_roc(model.predict_proba(X_test)[:,1], Y_test)