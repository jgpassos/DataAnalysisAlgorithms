# Libs

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import subprocess
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sklearn'])
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#---------------------------------------

data = pd.read_csv("basecarros.csv")

#data.info()

#print(data.columns)
#print(data.head())

data_binary = data[["origem", "mpg"]]
data_binary.columns = ["origem", "mpg"]

data_final = data_binary.fillna(method = "ffill")
#print("Dimentions {}".format(data_final.ndim))
#print("Size {}".format(data_final.size))

x = np.array(data_binary[["origem"]])#.reshape(-1,1)
y = np.array(data_binary["mpg"]).reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
train_score = regr.score(x_train, y_train)

print("O Score de treinamento do modelo é: ", train_score)
test_score = regr.score(x_test, y_test)
print("O Score de teste do modelo é: ", test_score)
y_pred = regr.predict(x_test)
print("Score da variância: %2.f" % r2_score(y_test, y_pred)) 
print("Coeficiente (Inclinação da Linha - b \n", regr.coef_)

print("O erro médio Quadrado: %.2f" % mean_squared_error(y_test, y_pred))

plt.scatter(x, y, color = 'b')
plt.xlabel("origem", fontsize=14)
plt.ylabel("miles per gallon", fontsize=14)
plt.title("AUTO MPG CAR", fontsize=17)
plt.plot(x_test, y_pred, color="k")

plt.show()