#import general modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import dataset and merge the test and training sets for
#                                   cleaning data at once(run once)
b = pd.read_csv("train_housing.csv")
#taking the dependent value from training set and dropping it to merge
y = b.iloc[:, -1].values

#getting y_test
temp = pd.read_csv('sample_submission.csv')
y_test = temp.iloc[:, 1].values

#Getting the dataset
dataset = pd.read_csv('final_output.csv',
                      header = 0,
                      na_values = 'nan')

#Dropping datasets that have too many NaN values/are of no use
drop_set = ['PoolQC','Fence','MiscFeature','FireplaceQu','Id','Alley']
dataset = dataset.drop(drop_set, axis=1)

#Dividing dataset into text and numeric
l = dataset.columns
dataset_number = pd.DataFrame()
dataset_strings = pd.DataFrame()
for i in range(74):
    temp_dataset = dataset[l[i]]    
    if (np.issubdtype(dataset[l[i]].dtype, np.number)) == True:
        dataset_number = pd.concat([dataset_number, temp_dataset], axis=1)
    else:
        dataset_strings = pd.concat([dataset_strings, temp_dataset], axis=1)            
    
#Checking the null values in both datasets
dataset_number.isnull().sum()
dataset_strings.isnull().sum()

#Cleaning up null values in numeric dataset
#Recovering missing data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
dataset_number = imp.fit_transform(dataset_number)
dataset_number = pd.DataFrame(dataset_number)

#Cleaning up null values in text dataset
#Recovering missing data (NaN)
dataset_strings = dataset_strings.apply(lambda x:x.fillna(x.value_counts().index[0]))

#Encoding the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dataset_strings = dataset_strings.apply(LabelEncoder().fit_transform)

#Since Data has label encoded for more than binary, we need to hot encode
for i in range(38):
    onehotencoder = OneHotEncoder(categorical_features = [i])
    dataset_strings = onehotencoder.fit_transform(dataset_strings).toarray()
dataset_strings = pd.DataFrame(dataset_strings)



#concatenating both the datasets to get a final cleaned dataset
cleaned_dataset = pd.concat([dataset_number, dataset_strings], axis=1)

#Dividing into test and train datasets
X_train = cleaned_dataset[0:1460]
X_test = cleaned_dataset[1460:2919]


# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 60, random_state = 0)
regressor.fit(X_train, y)

# Predicting a new result
y_pred = regressor.predict(X_test)

#To score the model
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

test_score = r2_score(y_test, y_pred)

predicted_train = regressor.predict(X_train)
predicted_test = regressor.predict(X_test)
spearman = spearmanr(y_test, predicted_test)
pearson = pearsonr(y_test, predicted_test)