import numpy as np  
from scipy.optimize import curve_fit  
import math
import pandas as pd
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

t = np.array([0, 1, 2, 3])  

#Logistic Model
def logistic_model(t, YM, Y0, K):  
    N = YM*Y0/((YM-Y0)*np.exp(-K*t) + Y0)
    return N  

data = pd.read_csv('E:\\resources\\work_related\\growth curve\\test\\Confluent_all.csv',header=0)
features = data[['day0', 'day1','day2','day3']]
target = data[['day4','day5','day6','day7']] 
seed = 1234

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=seed)
X_test_list = []  
for index, row in X_test.iterrows():  
    X_test_smaple = row.values.reshape(1, -1) 
    X_test_smaple = X_test_smaple.flatten()
    X_test_list.append(X_test_smaple)  
y_pred_list = []

for N in X_test_list:
    popt, pcov = curve_fit(logistic_model, t, N)  
    YM_value, Y0_value, K_value = popt 
    y_test_day4 = logistic_model(4, YM_value, Y0_value, K_value)
    y_test_day5 = logistic_model(5, YM_value, Y0_value, K_value)
    y_test_day6 = logistic_model(6, YM_value, Y0_value, K_value)
    y_test_day7 = logistic_model(7, YM_value, Y0_value, K_value)
    y_test_day = [y_test_day4, y_test_day5, y_test_day6, y_test_day7]
    y_pred_list.append(y_test_day)
y_pred = pd.DataFrame(y_pred_list, columns=["day4", "day5", "day6", "day7"])

mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)
print('Root Mean Squared Error:', rmse)
print('R^2 Score:', r2)

y_pred_logistic = y_pred
y_test = pd.DataFrame(y_test)
y_test = y_test.reset_index()
y_test_change = y_test.drop(columns=['index'])

list_y_test = []

for index, row in y_test_change.iterrows():
    list_y_test.append(row.values)

list_y_pred = []
for index, row in y_pred.iterrows():
    list_y_pred.append(row.values)   
from sklearn.metrics import mean_absolute_error
list_mre = []
for i in range(0,len(list_y_pred)):
    mae = mean_absolute_error(list_y_test[i],list_y_pred[i])
    relative_error = mae / (sum(list_y_pred[i]) / len(list_y_pred[i])) + mae / (sum(list_y_test[i]) / len(list_y_test[i]))
    relative_error_mean = relative_error / 2
    list_mre.append(relative_error_mean)
print(len(list_mre))
list_mre_new1 = [x for x in list_mre if x <= 0.1]
print(len(list_mre_new1))
list_mre_new2 = [x for x in list_mre if x > 0.1 and x <= 0.2]
print(len(list_mre_new2))
list_mre_new3 = [x for x in list_mre if x > 0.2 and x <= 0.4]
print(len(list_mre_new3))
list_mre_new4 = [x for x in list_mre if  x > 0.4]
print(len(list_mre_new4))

#Gompertz Model
def Gompertz_model(t, YM, Y0, K):  
    N = YM *(Y0/YM)**(np.exp(-K*t))
    return N 

for N in X_test_list:
    popt, pcov = curve_fit(Gompertz_model, t, N, maxfev=50000)  
    YM_value, Y0_value, K_value = popt 
    y_test_day4 = Gompertz_model(4, YM_value, Y0_value, K_value)
    y_test_day5 = Gompertz_model(5, YM_value, Y0_value, K_value)
    y_test_day6 = Gompertz_model(6, YM_value, Y0_value, K_value)
    y_test_day7 = Gompertz_model(7, YM_value, Y0_value, K_value)
    y_test_day = [y_test_day4, y_test_day5, y_test_day6, y_test_day7]
    y_pred_list.append(y_test_day)
y_pred = pd.DataFrame(y_pred_list, columns=["day4", "day5", "day6", "day7"])  

mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)
print('Root Mean Squared Error:', rmse)
print('R^2 Score:', r2)

y_test = pd.DataFrame(y_test)
y_test = y_test.reset_index()
y_test_change = y_test.drop(columns=['index'])

list_y_test = []

for index, row in y_test_change.iterrows():
    list_y_test.append(row.values)

list_y_pred = []
for index, row in y_pred.iterrows():
    list_y_pred.append(row.values)
    
from sklearn.metrics import mean_absolute_error
list_mre = []
for i in range(0,len(list_y_pred)):
    mae = mean_absolute_error(list_y_test[i],list_y_pred[i])
    relative_error = mae / (sum(list_y_pred[i]) / len(list_y_pred[i])) + mae / (sum(list_y_test[i]) / len(list_y_test[i]))
    relative_error_mean = relative_error / 2
    list_mre.append(relative_error_mean)
print(len(list_mre))
list_mre_new1 = [x for x in list_mre if x <= 0.1]
print(len(list_mre_new1))
list_mre_new2 = [x for x in list_mre if x > 0.1 and x <= 0.2]
print(len(list_mre_new2))
list_mre_new3 = [x for x in list_mre if x > 0.2 and x <= 0.4]
print(len(list_mre_new3))
list_mre_new4 = [x for x in list_mre if  x > 0.4]
print(len(list_mre_new4))


