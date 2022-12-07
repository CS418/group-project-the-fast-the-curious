# CS418 - Data Science Project
# Group Name: The Fast & The Curious
# DATE: Dec 6, 2022

# Group Members: 
# Georgi Nikolov    (gnikol5@uic.edu)  (Github: https://github.com/NikolovG)
# Daniel Valencia   (dvalen2@uic.edu)  (Github: https://github.com/Valencia24)
# Lizbeth Gutierrez (igutie37@uic.edu) (Github: https://github.com/lgutie37)
# Jovad Uribe       (juribe5@uic.edu)  (Github: https://github.com/jovuribe)


import pandas as pd
import requests

url = "https://meteostat.p.rapidapi.com/stations/daily"
querystring = {"station":"KPWK0","start":"2012-01-01","end":"2021-01-01","model":"true","units":"imperial"}
headers = {
	"X-RapidAPI-Key": "6fec950325msha54f5bc890b65e0p1fd8d0jsnc882976e1304",
	"X-RapidAPI-Host": "meteostat.p.rapidapi.com"
}
response = requests.request("GET", url, headers=headers, params=querystring)
json = response.json()
type(json['data'])
type(json['data'][0])
df = pd.DataFrame(json['data'])
df

# we will be keeping date, tavg, tmin, tmax, prcp, and pres columns, here we drop the columns discussed in the markdown above:
df = df.drop("snow", axis='columns')
df = df.drop("wdir", axis='columns')
df = df.drop("wspd", axis='columns')
df = df.drop("wpgt", axis='columns')
df = df.drop("tsun", axis='columns')

# shows the rows with NaN in rows:
df[df['prcp'].isna()]
df[df['pres'].isna()]

# drop rows with NaN in rows:
df = df.dropna(subset=['prcp','pres'])

# print post dropped rows, after cleaning data
df
df.describe()
df.info()

import numpy as np
import matplotlib.pyplot as plt

columns = np.array(df.columns)
for count, column in enumerate(columns[1:]):
    plt.subplot(2, 3, count+1)
    plt.plot(df[column])
    plt.title(column)
plt.show()
df.boxplot(column=['tavg', 'tmin', 'tmax'])  
df.boxplot(column=['prcp'])
df.boxplot(column=['pres'])

# pip install statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_check_values = df[['tavg', 'tmin', 'tmax']].dropna()

vif_df = pd.DataFrame()
vif_df["VIF"] = [variance_inflation_factor(vif_check_values.values, i) for i in range(len(vif_check_values.columns))]
vif_df

df[['tavg', 'tmin', 'tmax']].plot()

# fig = plt.figure()
askPlot = plt.axes(projection='3d')
x = df["pres"]
z = df["tmin"]
y = df["tmax"]
askPlot.scatter(x, y, z, c=z, cmap='viridis')

df[["pres", "tmax"]].corr()

# pip install meteostat
from datetime import datetime
from meteostat import Point, Daily

# Set time period
start = datetime(2021, 1, 1)
end = datetime(2021, 12, 31)

# Create Point for Chicago
chicago = Point(41.868755, -87.646090, 70)

# Get daily data for 2018
data = Daily(chicago, start, end)
data = data.fetch()

# Plot line chart including average, minimum and maximum temperature
data.plot(y=['tavg', 'tmin', 'tmax'])
df.hist()
plt.title("Temperature Over 2021")
plt.show()

# Set time period
start = datetime(1995, 1, 1)
end = datetime(1995, 12, 31)

# Create Point for Chicago
chicago = Point(41.868755, -87.646090, 70)

# Get daily data for 2018
data = Daily(chicago, start, end)
data = data.fetch()

# Plot line chart including average, minimum and maximum temperature
data.plot(y=['tavg', 'tmin', 'tmax'])
data.hist()
plt.title("Temperature Over 1995")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import matplotlib.pyplot as plt

# remove dashes from date column and convert the column type to an int
df['date'] = df['date'].replace('-', '', regex=True).astype(int)

# drop rows with NaN in rows:
df = df.dropna(subset=['tavg','tmin','tmax'])

# split the data into two sets
X = df.drop(["prcp"], axis=1)
Y = df["prcp"]

# create training and testing data using a 80/20 training/test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# create decision tree model and train it
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)

# make predictions on test set
y_pred = tree.predict(X_test)

# evaluating decision tree model
# Reference: https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print("DecisionTreeRegressor Metrics:")
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)

# plot the predicted and actual values
# Reference: https://www.datatechnotes.com/2020/10/regression-example-with-decisiontreeregressor.html
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=1, label="Actual")
plt.plot(x_ax, y_pred, linewidth=1.1, label="Predicted")
plt.title("Predicting Precipitation Using DecisionTreeRegressor")
plt.ylabel('Precipitation Values in mm')
plt.legend(loc='upper right',fancybox=True, shadow=True)
plt.grid(True)
plt.show()

# create random forest model and train it
forest = RandomForestRegressor(random_state=42)
forest.fit(X_train, y_train)

# make predictions on test set
y_pred = forest.predict(X_test)

# evaluating random forest model
# Reference: https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print("RandomForestRegressor Metrics:")
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)

# plot the predicted and actual values
# Reference: https://www.datatechnotes.com/2020/10/regression-example-with-decisiontreeregressor.html
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=1, label="Actual")
plt.plot(x_ax, y_pred, linewidth=1.1, label="Predicted")
plt.title("Predicting Precipitation Using RandomForestRegressor")
plt.ylabel('Precipitation Values in mm')
plt.legend(loc='upper right',fancybox=True, shadow=True)
plt.grid(True)
plt.show()

# example prediction from random day in dataset
# Reference: https://www.samloves.coffee/rainML 
pred_day = X.loc[X['date'] == 20190523]
pred_val = round(forest.predict(pred_day)[0], 3)


actual_day = df.loc[df['date'] == 20190523]
actual_val = round(actual_day.iloc[0]["prcp"], 3)

print("Predicted precipitation: ", pred_val)
print("Actual precipitation: ", actual_val)

# using bar chart to visualize model prediction
vals = {"Values":["Predicted", "Actual"], "in mm":[pred_val, actual_val]}
bc = pd.DataFrame(data=vals)
axes = bc.plot.bar(x="Values", y="in mm", title="Comparing Precipitation Results for 05/23/2019");
axes.legend(loc=9)

# pip install prophet
from datetime import datetime
import math
from meteostat import Point
from meteostat import Daily
import pandas as pd

from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Set time period
startDate = datetime(2021, 1, 1)
endDate = datetime(2022, 9, 30)

# Create Point for Chicago
chicago = Point(41.868755, -87.646090, 70)
data = Daily(chicago, startDate, endDate)
data = data.fetch()

# Setting up training and testing data
train = data.loc[:'2022-08-31']
test = data.loc['2022-09-01':]

train = data[['tavg']]
train = train.reset_index()
train.columns = ['ds', 'y']

# Establishing our ML model 
model = Prophet()
model.fit(train)
future = pd.DataFrame(test.index.values)
future.columns = ['ds']
forecast = model.predict(future)

y_true = test['tavg'].values
y_pred = forecast['yhat'].values
diff = math.sqrt(mean_squared_error(y_true, y_pred))
print('Diff. Betwwen Exp & Actual :', diff)

# plot expected vs actual
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.ylim(ymax=30, ymin=15)
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression

# Reference: https://stackabuse.com/linear-regression-in-python-with-scikit-learn/
# split datasets
y = df['tmax'].values.reshape(-1, 1)
X = df['pres'].values.reshape(-1, 1)

# get training and test sets based on 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# create linear regression model and train it
linear = LinearRegression()
linear.fit(X_train, y_train)

# make predictions on test set
y_pred = linear.predict(X_test)

# evaluate linear regression model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print("Linear Regression Metrics:")
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)

# split datasets
y = df['tmax']
X = df[['pres', 'prcp']]

# get training and test sets based on 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# create multilinear regression model and train it
multilinear = LinearRegression()
multilinear.fit(X_train, y_train)

# displaying how each feature affects the max temperature
feature_names = X.columns
model_coeffs = multilinear.coef_
coeffs_df = pd.DataFrame(data = model_coeffs, index = feature_names, columns = ['Coefficient value'])

# make predictions on test set
y_pred = multilinear.predict(X_test)

# evaluate linear regression model
mae2 = mean_absolute_error(y_test, y_pred)
mse2 = mean_squared_error(y_test, y_pred)
rmse2 = math.sqrt(mse2)

print(coeffs_df)
print("Linear Regression Metrics:")
print("Mean Absolute Error: ", mae2)
print("Mean Squared Error: ", mse2)
print("Root Mean Squared Error: ", rmse2)

# metrics plotted for each model to visually show similarities
# Reference: https://www.geeksforgeeks.org/create-a-grouped-bar-plot-in-matplotlib/
import numpy as np
x = np.arange(2)
y1 = [15.72, 15.73]
y2 = [357.42, 359.31]
y3 = [18.91, 18.96]

plt.bar(x-0.2, y1, 0.2, color='red')
plt.bar(x, y2, 0.2, color='green')
plt.bar(x+0.2, y3, 0.2, color='blue')
plt.xticks(x, ['Linear', 'Multi Linear'])
plt.xlabel("Regression Models")
plt.ylabel("Metrics")
plt.legend(["MAE", "MSE", "RMSE"], loc=9)
plt.show()

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = df.dropna()
  
hyperparam = auto_arima(df['tavg'], seasonal = True, start_p = 1, start_q = 1, max_p = 3, max_q = 3, m = 12, start_P = 0, d = 1, D = 1, error_action ='ignore', suppress_warnings = True, stepwise = True)
  
hyperparam.summary()


train = df.iloc[:len(df)-316]
test = df.iloc[len(df)-316:]
  
model = SARIMAX(df['tavg'], seasonal_order =(2, 1, [], 12))
  
fitted_model = model.fit()
fitted_model.summary()
  
predictions = fitted_model.predict(len(train), (len(train)+len(test)-1), yp = 'levels').rename("Predictions")
  
predictions.plot(legend = True)
test['tavg'].plot(legend = True)

mae = mean_absolute_error(test['tavg'], predictions)
mse = mean_squared_error(test['tavg'], predictions)
rmse = math.sqrt(mse)

print("ARIMA Model Metrics:")
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", rmse)