import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

predict = 'cnt'

# get the data
data = pd.read_csv('./hour.csv')
data = data[['season', 'yr', 'mnth', 'hr', 'temp', 'workingday', 'hum', 'windspeed', 'weathersit', 'cnt']]
x = np.array(data.drop([predict], 1))  # the data to make prediction
y = np.array(data[predict])  # the counts of people who rode a bike

# get train and test sets of data for x and y
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

model = LinearRegression().fit(x_train, y_train)
x_pred = model.predict(x_test)
acc = model.score(x_test, y_test)
print(acc)
