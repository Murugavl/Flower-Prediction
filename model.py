import pandas as pd
import numpy as np
import pickle

from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.svm import SVC
model = SVC(kernel='linear').fit(X_train,y_train)

pickle.dump(model, open('iris.pkl','wb'))

