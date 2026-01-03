import numpy as np 

class LinearRegression:

    def __init__(self,lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        # init parameters
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0


        #gradient descent or finding the gradient

        for _ in range(self.n_iters):
            # y = Wx + b
            y_predicted = np.dot(X, self.weights) + self.bias

            # print(f'y_predicted is {y_predicted}')

            #derivative in respect to the weight
            dw = (1/n_samples) * np.dot((X.T), (y_predicted-y))

            #derivative in respect to the bias
            db = (1/n_samples) * np.sum(y_predicted - y)

            # update weight
            self.weights -= self.lr * dw

            # print(f'updated weight is {self.weights}')
            self.bias -= self.lr * db

            # print(f'updated bias is {self.bias}')

    def predict(self,X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
