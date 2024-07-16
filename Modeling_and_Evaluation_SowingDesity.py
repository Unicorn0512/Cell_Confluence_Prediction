import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import scipy.stats as stats
from pprint import pprint
from sklearn import metrics
from openpyxl import load_workbook
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('E:\\resources\\work_related\\growth curve\\test\\Confluent_conc_all.csv',header=0)
len(data)
seed = 1234
features = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7,24,25,26,27,28,29,30,31]]
target = data.iloc[:, [8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=seed)

n_estimators_range=[int(x) for x in np.linspace(start=10,stop=3000,num=60)]
max_features_range=['auto','sqrt']
max_depth_range=[int(x) for x in np.linspace(10,500,num=50)]
max_depth_range.append(None)
min_samples_split_range=[2,5,10]
min_samples_leaf_range=[1,2,4,8]
bootstrap_range=[True,False]
random_forest_hp_range={'n_estimators':n_estimators_range,
                        'max_features':max_features_range,
                        'max_depth':max_depth_range,
                        'min_samples_split':min_samples_split_range,
                        'min_samples_leaf':min_samples_leaf_range
                        # 'bootstrap':bootstrap_range
                        }
random_forest_model_test_base=RandomForestRegressor()
random_forest_model_test_random=RandomizedSearchCV(estimator=random_forest_model_test_base,
                                                   param_distributions=random_forest_hp_range,
                                                   n_iter=200,
                                                   n_jobs=-1,
                                                   cv=5,
                                                   verbose=1,
                                                   random_state=seed
                                                   )
random_forest_model_test_random.fit(X_train, y_train)
best_hp_now=random_forest_model_test_random.best_params_
best_hp_now
random_forest_hp_range_2={'n_estimators':[1910,1950,1986,2020,2060],
                          'max_features':[2,3],
                          'max_depth':[260,290,320,350],
                          'min_samples_split':[2,3] ,
                          'min_samples_leaf':[1,2]
                          }
random_forest_model_test_2_base=RandomForestRegressor()
random_forest_model_test_2_random=GridSearchCV(estimator=random_forest_model_test_2_base,
                                               param_grid=random_forest_hp_range_2,
                                               cv=5,
                                               verbose=1,
                                               n_jobs=-1)
random_forest_model_test_2_random.fit(X_train, y_train)
best_hp_now_2=random_forest_model_test_2_random.best_params_
pprint(best_hp_now_2)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=seed)
regressor = RandomForestRegressor(n_estimators=2060, min_samples_split=2, min_samples_leaf=1, max_features=3, max_depth=290, random_state=seed)
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
rmse = sqrt(mse)
r2 = r2_score(y_train, y_train_pred)
print('Root Mean Squared Error:', rmse)
print('R^2 Score:', r2)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)
print('Root Mean Squared Error:', rmse)
print('R^2 Score:', r2)

y_test = y_test.reset_index()
y_test.head(5)
y_test_change = y_test.drop(columns=['index'])
y_test_change.head(5)
y_pred = pd.DataFrame(y_pred)
y_pred.head(5)
import pandas as pd
list_y_test = []
for index, row in y_test_change.iterrows():
    list_y_test.append(row.values)
list_y_pred = []
for index, row in y_pred.iterrows():
    list_y_pred.append(row.values)
len(list_y_pred)

list_mre = []
for i in range(0,len(list_y_pred)):
    mae = mean_absolute_error(list_y_test[i],list_y_pred[i])
    relative_error = mae / (sum(list_y_pred[i]) / len(list_y_pred[i])) + mae / (sum(list_y_test[i]) / len(list_y_test[i]))
    relative_error_mean = relative_error / 2
    list_mre.append(relative_error_mean)
print(len(list_mre))
list_mre_new = [x for x in list_mre if x <= 0.08]
print(len(list_mre_new))
list_mre_new = [x for x in list_mre if x > 0.08 and x <= 0.2]
print(len(list_mre_new))
list_mre_new = [x for x in list_mre if x > 0.2 and x <= 0.4]
print(len(list_mre_new))
list_mre_new = [x for x in list_mre if  x > 0.4]
print(len(list_mre_new))

#only 1600 cell/well
seed = 1234
features = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7,24,25,26,27,28,29,30,31]]
target = data.iloc[:, [8,9,10,11,12,13,14,15]]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=seed)
regressor = RandomForestRegressor(n_estimators=2060, min_samples_split=2, min_samples_leaf=1, max_features=3, max_depth=290, random_state=seed)
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)

mse = mean_squared_error(y_train, y_train_pred)
rmse = sqrt(mse)
r2 = r2_score(y_train, y_train_pred)
print('Root Mean Squared Error:', rmse)
print('R^2 Score:', r2)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)
print('Root Mean Squared Error:', rmse)
print('R^2 Score:', r2)

y_test = pd.DataFrame(y_test)
y_test = y_test.reset_index()
y_test_change = y_test.drop(columns=['index'])
y_pred = pd.DataFrame(y_pred)

list_y_test = []
for index, row in y_test_change.iterrows():
    list_y_test.append(row.values)
list_y_pred = []
for index, row in y_pred.iterrows():
    list_y_pred.append(row.values)    
list_r2 = []
for i in range(0,len(list_y_pred)):
    r2 = r2_score(list_y_test[i],list_y_pred[i])
    list_r2.append(r2)
print(len(list_r2))
list_r2_new = [x for x in list_r2 if x > 0.9]
print(len(list_r2_new))
list_r2_new = [x for x in list_r2 if x > 0.8 and x <= 0.9]
print(len(list_r2_new))
list_r2_new = [x for x in list_r2 if x > 0.6 and x <= 0.8]
print(len(list_r2_new))
list_r2_new = [x for x in list_r2 if  x <= 0.6]
print(len(list_r2_new))

list_mre = []
for i in range(0,len(list_y_pred)):
    mae = mean_absolute_error(list_y_test[i],list_y_pred[i])
    relative_error = mae / (sum(list_y_pred[i]) / len(list_y_pred[i])) + mae / (sum(list_y_test[i]) / len(list_y_test[i]))
    relative_error_mean = relative_error / 2
    list_mre.append(relative_error_mean)
print(len(list_mre))
list_mre_new = [x for x in list_mre if x <= 0.08]
print(len(list_mre_new))
list_mre_new = [x for x in list_mre if x > 0.08 and x <= 0.2]
print(len(list_mre_new))
list_mre_new = [x for x in list_mre if x > 0.2 and x <= 0.4]
print(len(list_mre_new))
list_mre_new = [x for x in list_mre if  x > 0.4]
print(len(list_mre_new))

#only 800 cell/well
seed = 1234
features = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7,24,25,26,27,28,29,30,31]]
target = data.iloc[:, [16,17,18,19,20,21,22,23]]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=seed)
regressor = RandomForestRegressor(n_estimators=2060, min_samples_split=2, min_samples_leaf=1, max_features=3, max_depth=290, random_state=seed)
regressor.fit(X_train, y_train)

y_train_pred = regressor.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
rmse = sqrt(mse)
r2 = r2_score(y_train, y_train_pred)
print('Root Mean Squared Error:', rmse)
print('R^2 Score:', r2)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)
print('Root Mean Squared Error:', rmse)
print('R^2 Score:', r2)

y_test = pd.DataFrame(y_test)
y_test = y_test.reset_index()
y_test_change = y_test.drop(columns=['index'])
y_pred = pd.DataFrame(y_pred)
list_y_test = []
for index, row in y_test_change.iterrows():
    list_y_test.append(row.values)
list_y_pred = []
for index, row in y_pred.iterrows():
    list_y_pred.append(row.values)   
list_r2 = []
for i in range(0,len(list_y_pred)):
    r2 = r2_score(list_y_test[i],list_y_pred[i])
    list_r2.append(r2)

print(len(list_r2))
list_r2_new = [x for x in list_r2 if x > 0.9]
print(len(list_r2_new))
list_r2_new = [x for x in list_r2 if x > 0.8 and x <= 0.9]
print(len(list_r2_new))
list_r2_new = [x for x in list_r2 if x > 0.6 and x <= 0.8]
print(len(list_r2_new))
list_r2_new = [x for x in list_r2 if  x <= 0.6]
print(len(list_r2_new))

from sklearn.metrics import mean_absolute_error
list_mre = []
for i in range(0,len(list_y_pred)):
    mae = mean_absolute_error(list_y_test[i],list_y_pred[i])
    relative_error = mae / (sum(list_y_pred[i]) / len(list_y_pred[i])) + mae / (sum(list_y_test[i]) / len(list_y_test[i]))
    relative_error_mean = relative_error / 2
    list_mre.append(relative_error_mean)
print(len(list_mre))


