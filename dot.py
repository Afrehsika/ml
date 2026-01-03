import numpy as np

a = np.array([1,2])
b = np.array([3,4])

def dot(x,y):
    ans = 0 
    for i in range(len(x)):
        ans += x[i] * y[i]
    return ans

print(dot(a,b))