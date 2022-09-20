# Libs

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import subprocess
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sklearn'])
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#---------------------------------------

data = pd.read_csv("wpbc2.csv")

data.info()

print(data.columns)
print(data.head())

print(data.isnull().values.any())

data = data.fillna(method = "ffill")
print("Dimentions {}".format(data.ndim))
print("Shape {}".format(data.shape))
print("Size {}".format(data.size))

print(data.describe())

x = np.array(data[["texture_worst"]])#.reshape(-1,1)
y = np.array(data["area_worst"]).reshape(-1,1)

print("Shape de x")
print(x.shape)
print(y.shape)

#- split the dataset into 2 sets: training and test

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
train_score = regr.score(x_train, y_train)
print("O Score de treinamento do modelo é: ", train_score)
test_score = regr.score(x_test, y_test)
print("O Score de teste do modelo é: ", test_score)
y_pred = regr.predict(x_test)

print("Score da variância: %2.f" % r2_score(y_test, y_pred)) 
print("Coeficiente (Inclinação da Linha) \n", regr.coef_)
print("O erro médio Quadrado: %.2f" % mean_squared_error(y_test, y_pred))

plt.scatter(x, y, color='red')
plt.xlabel("texture_worst", fontsize=14)
plt.ylabel("area_worst", fontsize=14)
plt.title("Breast Cancer Wisconsin: Texture worst vs Area Worst", fontsize=17)
plt.plot(x_test, y_pred, color="blue")

plt.show()