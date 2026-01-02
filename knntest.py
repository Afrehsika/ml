import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#FF0000",'#00FF00','#0000FF'])

iris = datasets.load_iris()
X,y = iris.data, iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=1234)

print(X_test.shape)

plt.figure(figsize=(10,6))
plt.scatter(X_test[:,0],X_test[:,1], c=y_test, cmap=cmap, edgecolor='k', s=100 )
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Test Data Distribution")
plt.show()