from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("C:\Users\rbgus\Desktop\test1", "rb")
df.head()
X = df["X"]
y = df["Y"]
# m, b = np.polyfit(X, y, 1)
plt.plot(X, y, 'o')
# plt.plot(X, m*X + b)
line_fitter = LinearRegression()
line_fitter.fit(X.values.reshape(-1,1), y)
plt.plot(X,line_fitter.predict(X.values.reshape(-1,1)))


print('기울기 a : ', line_fitter.coef_)
print('y절편 : ', line_fitter.intercept_)

plt.show()
# 기울기


