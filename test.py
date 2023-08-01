from sklearn.linear_model import LinearRegression

X, y = [[1],[2]], [10, 20]
lr = LinearRegression(fit_intercept = False)
lr.fit(X, y)
print(lr.predict([[3], [4]]))