import pickle

import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression


def preprocess(name, label_encoder, data_set):
    """A little preprocess function I made to process the data and then set it back to the original data"""
    new = label_encoder.fit_transform(list(data_set[name]))
    data_set[name] = new


def show_graph(dat, p):
    style.use("ggplot")
    pyplot.scatter(dat[p], dat['G3'])
    pyplot.xlabel(p)
    pyplot.ylabel('Final Grade')
    pyplot.show()


def train_model(x, y):
    best = 0
    for _ in range(5000):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

        model = LinearRegression()
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)
        print(acc)

        if acc > best:
            best = acc
            with open('model.pickle', 'wb') as f:
                pickle.dump(model, f)

    print(f'best was: {best}')


def load_model():
    pickle_in = open('model.pickle', 'rb')
    return pickle.load(pickle_in)


# read the csv and setup the data
filename = 'student-mat.csv'
data = pd.read_csv(filename, sep=';')
data = data[['school', 'age', 'famsize', 'schoolsup', 'famsup', 'traveltime', 'higher', 'internet', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']]

le = preprocessing.LabelEncoder()
preprocess('school', le, data)
preprocess('famsize', le, data)
preprocess('schoolsup', le, data)
preprocess('famsup', le, data)
preprocess('higher', le, data)
preprocess('internet', le, data)

predict = 'G3'
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

# training the model (runs 5k times to get the best accuracy, then saves the best one)
# train_model(x, y)

# load the model regularly from the file
model = load_model()

show_graph(data, 'G1')
show_graph(data, 'G2')

pred = model.predict(x_test[0].reshape(1, -1))
# print(pred)  # the predicted test score for the above student
