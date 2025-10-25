import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Step-1 Generate Toy(Dummy) Dataset

X,y = make_blobs(n_samples=2000, n_features=2,cluster_std=3, centers=2, random_state=42)
n_features=2
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
    loss= -np.mean(y*np.log(yp)+(1-y)*np.log(1-yp))
    return loss


def gradient(X,y,yp):
    m=X.shape[0]
    grad=-(1/m)*np.dot(X.T,(y-yp))
    return grad

def train(X,y,max_iters=100,learning_rate=0.1):
    theta = np.random.randn(n_features+1,1)
    error_list = []
    for i in range(max_iters):
       yp= hypothesis(X,theta)
       e= error(y,yp)
       error_list.append(e)
       grad= gradient(X,y,yp)
       theta= theta - learning_rate*grad

    plt.plot(error_list)
    plt.show()
    return theta


def predict(X,theta):
    h= hypothesis(X,theta)
    predis= np.zeros((X.shape[0],1),dtype="int")
    predis[h>0.5]=1

    return predis

def accuracy(X,y,theta):
    predis=predict(X,theta)
    return ((y==predis).sum())/X.shape[0]*100


def addExtraColumns(X):
    if X.shape[1]== n_features:
        ones=np.ones((X.shape[0],1))
        X = np.hstack((ones,X))

    return X


XT = addExtraColumns(XT)
print(XT)

Xt=addExtraColumns(Xt)

yT=yT.reshape(-1,1)
yt=yt.reshape(-1,1)
theta = train(XT,yT,max_iters=300,learning_rate=0.2)
print(theta)

print("Learned theta\n" ,  theta)

plt.scatter(XT[:,1], XT[:,2], c=yT,cmap="viridis")
#plt.show()

x1 = np.linspace(-3,3,6)
x2=-(theta[0]+theta[1]*x1)/theta[2]
plt.plot(x1,x2)
plt.show()

predict(Xt,theta)
aT= accuracy(XT,yT,theta)
at= accuracy(Xt,yt,theta)
print(at)


# now using library
model= LogisticRegression()
X,y = make_blobs(n_samples=2000, n_features=2,cluster_std=3, centers=2, random_state=42)
model.fit(X,y)

model.predict(X)

model.score(X,y)

#Multi class

X,y =make_blobs(n_samples=2000, n_features=2,cluster_std=3, centers=2, random_state=42)
plt.scatter(X[:,0], X[:,1], c=y,cmap="viridis")
plt.show()

model= LogisticRegression()
model.fit(X,y)
model.predict(X)
model.score(X,y)