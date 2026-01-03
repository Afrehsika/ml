import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X,y = datasets.make_regression(n_samples=100, n_features=2, noise=30 ,random_state=8)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=45)
 
# fig, axes = plt.subplots(1,2)

# # Feature 1 vs y
# axes[0].scatter(X[:, 0], y, marker="o", s=30)
# axes[0].set_xlabel("X[:, 0]")
# axes[0].set_ylabel("y")
# axes[0].set_title("Feature 1 vs Target")

# # Feature 2 vs y
# axes[1].scatter(X[:, 1], y, marker="o", s=30)
# axes[1].set_xlabel("X[:, 1]")
# axes[1].set_ylabel("y")
# axes[1].set_title("Feature 2 vs Target")


# plt.tight_layout()
# plt.show()

from linear_regression import LinearRegression

test1 = LinearRegression(lr=0.01)
test1.fit(X_train,y_train)
predict = test1.predict(X_test)


def mse(y_true,y_predicted):
    return np.mean((y_true-y_predicted)**2)


mse_value = mse(y_test,predict)
print(mse_value)

cmap = plt.get_cmap("viridis")
plt.figure(figsize=(10,6))
plt.scatter(y_test, predict, color=cmap(0.6), edgecolor='k', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()