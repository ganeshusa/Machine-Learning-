import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

#Step-1 Generate Data
X, y = make_regression(n_samples=500, n_features=10, n_informative=5, noise=25.8, random_state=0)
print(X.shape, y.shape)
n_features = X.shape[1]

pd.DataFrame(X).head()


#def normalize
def normalize(X):
   u=X.mean(axis=0)
   std=X.std(axis=0)
   return (X-u)/std



X=normalize(X)

#visulization
for f in range(0,9):
    plt.subplot(3,3,f+1)
    plt.scatter(X[:,f],y)
plt.show()

#train test split
XT,Xt,yT,yt = train_test_split(X, y, test_size=0.3,shuffle=False,
                                                    random_state=0)
print(XT.shape,yT.shape,Xt.shape,yt.shape)

def preprocess(X):
    #add a common column of 1s in X as oth colum
    if X.shape[1]==n_features:
        m = X.shape[0]
        ones = np.ones((m, 1))
        X = np.hstack((ones, X))
    return X


def hypothesis(X, theta):
    return np.dot(X, theta)

def loss(X, y, theta):
    yp = hypothesis(X, theta)
    error = np.mean((yp-y)**2)
    return error

def gradient(X, y, theta):
    yp = hypothesis(X, theta)
    grad = np.dot(X.T, yp-y)
    m= X.shape[0]
    return grad/m


def train(X,y,learning_rate =0.1,max_iters=100):
    n=X.shape[1]
    theta = np.random.randn(n)
    error_list=[]
    for it in range(max_iters):
        e=loss(X,y,theta)
        error_list.append(e)
        grad = gradient(X, y, theta)
        theta = theta - learning_rate * grad  # Fixed update rule

    plt.plot(error_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.grid(True)
    plt.show()
    return theta

theta = train(XT,yT)
print(theta)

def r2_score(y, yp):
    ymean = y.mean()
    num = np.sum((y - yp) ** 2)
    denom = np.sum((y - ymean) ** 2)
    return 1 - num / denom

yp=hypothesis(Xt,theta)
r2 = r2_score(yt, yp)
print(r2)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(XT,yT)
yp= model.predict(Xt)

model.score(Xt,yt)
var = model.coef_
print(var)
model.intercept_



