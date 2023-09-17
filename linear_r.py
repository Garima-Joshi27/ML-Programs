 import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("placement.csv")
df.head()

plt.scatter(df["cgpa"], df["package"])
plt.xlabel("cgpa")
plt.ylabel("package in lpa")
# plt.show()

x = df.iloc[:, 0:1]
y = df.iloc[:, -1]
# print(y)
# print(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

# print(x_test)
# print(y_test)

lr.predict(x_test.iloc[0].values.reshape(1, 1))


plt.scatter(df["cgpa"], df["package"])
plt.plot(x_train, lr.predict(x_train), color="red")
plt.xlabel("CGPA")
plt.ylabel("Package(in lpa)")

plt.show()

# m = lr.coef_
# b = lr.intercept_
# print(m * 8.58 + b)
