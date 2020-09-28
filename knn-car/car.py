import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


def preprocess(name, label_encoder, data_set):
    new = label_encoder.fit_transform(list(data_set[name]))
    data_set[name] = new


data = pd.read_csv('./car.data')

le = preprocessing.LabelEncoder()
preprocess('buying', le, data)
preprocess('maint', le, data)
preprocess('doors', le, data)
preprocess('persons', le, data)
preprocess('lug_boot', le, data)
preprocess('safety', le, data)
preprocess('class', le, data)

predict = 'class'
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)

best = 0
acc = 0
for _ in range(1000):
    acc = model.score(x_test, y_test)
    if best <= acc:
        best = acc

print(acc)
