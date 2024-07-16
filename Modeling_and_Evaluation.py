import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('E:\\resources\\work_related\\growth curve\\test\\Confluent_all.csv', header = 0)

print(data.describe())
print(data[data.isnull()==True].count())
data.boxplot()
plt.savefig("boxplot.jpg")
plt.show()
print(data.corr())

features = data[['day0', 'day1','day2','day3']]
target = data[['day4','day5','day6','day7']] 
seed = 1234
#Random forest regression
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=seed)
regressor = RandomForestRegressor(n_estimators=500, random_state=seed)
regressor.fit(X_train, y_train)

regressor.feature_importances_
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R^2 Score:', r2)

y_train_pred = regressor.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
r2 = r2_score(y_train, y_train_pred)
print('Mean Squared Error:', mse)
print('R^2 Score:', r2)

y_test = y_test.reset_index()
y_test.head(5)
y_test_change = y_test.drop(columns=['index'])
y_test_change.head(5)
y_pred = pd.DataFrame(y_pred)
y_pred.head(5)

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
list_mre

list_mre_new1 = [x for x in list_mre if   x <= 0.1]
len(list_mre_new1)
list_mre_new2 = [x for x in list_mre if   x> 0.1 and x <= 0.2]
len(list_mre_new2)
list_mre_new3 = [x for x in list_mre if   x> 0.2 and x <= 0.4]
len(list_mre_new3)
list_mre_new4 = [x for x in list_mre if   x> 0.4]
len(list_mre_new4)

