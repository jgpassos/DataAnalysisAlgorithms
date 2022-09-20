import numpy as np
import pandas as pd
import sys
import subprocess
#subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sklearn'])
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree

#------------------------------------------------------

#Read dataset:

data = pd.read_csv("basecarros.csv")

#data.info()

#print(data.columns)
#print(data.head())

#Fill NAs:

data = data.fillna(method = "ffill")
print("Dimentions {}".format(data.ndim))
print("Shape {}".format(data.shape))
print("Size {}".format(data.size))

x = np.array(data[["potencia"]])
y = np.array(data["aceleracao"])

#Fit regression model:

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(x, y)
regr_2.fit(x, y)

#Predict 

x_test = np.arange(0, 200.0)[:, np.newaxis]
y_1 = regr_1.predict(x_test)
y_2 = regr_2.predict(x_test)

#Plot results:
plt.figure()
plt.scatter(x, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(x_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
plt.plot(x_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
