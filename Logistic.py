import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

#Step-1 Generate Toy(Dummy) Dataset

X,y = make_blobs(n_samples=2000, n_features=2,cluster_std=3, centers=2, random_state=42)
num_of_features=2
print(X.shape,y.shape)

#Setp -2 Visualise DataSet
def visualise(X,y):
    plt.scatter(X[:,0], X[:,1], c=y,cmap="viridis")
    plt.show()

#visualise(X,y)

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

X = normalize(X)
#visualise(X,y)


#setp -4 Train Test Split

XT,Xt,yT,yt=train_test_split(X,y,test_size=0.25,random_state=0)
#visualise(Xt,yt)
print(XT.shape,yT.shape)
print(yT.shape,yT.shape)

#Model
def sigmoid(z):
    return 1/(1+np.exp(-z))

def hypothesis(X,theta):
    return sigmoid(np.dot(X,theta))




z = np.linspace(-10,10,20)
plt.plot(z,sigmoid(z))
plt.show()


def error(y,yp):
    loss= -np.mean(y*(np.log(yp)+(1-y)*np.log(1-yp)))
    return loss


def gradient(X,y,yp):
    m=X.shape[0]
    grad=-(1/m)*np.dot(X.T,(y-yp))
    return grad

def train(X,y,max_iters=100,learning_rate=0.1):
    theta = np.random.randn(num_of_features+1,1)
    error_list = []
    for i in range(max_iters):
       yp= hypothesis(X,theta)
       e= error(y,yp)
       error_list.append(e)
       grad= gradient(X,y,yp)
       theta= theta - learning_rate*grad

    plt.plot(error_list)
    return theta


def addExtraColumns(X):
    if X.shape[1]== num_of_features:
        ones=np.ones((X.shape[0],1))
        X = np.hstack((X,ones))

    return X


XT = addExtraColumns(XT)
print(XT)

Xt=addExtraColumns(Xt)

yT=yT.reshape(-1,1)
yt=yt.reshape(-1,1)
theta = train(XT,yT,max_iters=300,learning_rate=0.2)
print(theta)

print(theta)
