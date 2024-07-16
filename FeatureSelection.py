import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data = pd.read_csv('E:\\resources\\work_related\\growth curve\\test\\Confluent_all.csv', header = 0)

features = data[['day0','day1','day2','day3']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores

features = data[['day0','day1','day2']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores)
print(average_score)
scores

features = data[['day0','day1','day3']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores

features = data[['day0','day2','day3']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores)
print(average_score)
scores

features = data[['day1','day2','day3']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores

features = data[['day0','day1']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores

features = data[['day0','day2']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores

features = data[['day0','day3']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores

features = data[['day1','day2']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores

features = data[['day1','day3']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores

features = data[['day2','day3']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores

features = data[['day0']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores

features = data[['day1']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores

features = data[['day2']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores

features = data[['day3']]
target = data[['day4','day5','day6','day7']] 
seed = 42
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
kfold = KFold(n_splits=5,random_state=seed)
scores = cross_val_score(regressor,features,target,cv = kfold,scoring = 'r2')
average_score = np.mean(scores) 
print(average_score)
scores
