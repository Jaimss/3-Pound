import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


def preprocess(name, label_encoder, data_set):
    new = label_encoder.fit_transform(list(data_set[name]))
    data_set[name] = new


# read the csv
data = pd.read_csv('student-mat.csv', sep=';')
data = data[['school', 'age', 'famsize', 'schoolsup', 'famsup',  'traveltime', 'higher', 'internet', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']]
print(data.head())

le = preprocessing.LabelEncoder()
preprocess('school', le, data)
preprocess('famsize', le, data)
preprocess('schoolsup', le, data)
preprocess('famsup', le, data)
preprocess('higher', le, data)
preprocess('internet', le, data)
print(data.head())

# grades are from 1-20
predict = 'G3'
x_train, x_test, y_train, y_test = model_selection.train_test_split(np.array(data.drop([predict], 1)), np.array(data[predict]), test_size=0.1)

model = LinearRegression().fit(x_train, y_train)
pred = model.predict(x_test)
acc = model.score(x_test, y_test)
print(acc)
