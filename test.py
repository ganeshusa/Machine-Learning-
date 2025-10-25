#hypothesis
import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns


def hypothesis(X,theta):
    return theta[0]+theta[1]*X

def error(X,y,theta):
    m=X.shape[0]
    e=0
    for i in range(m):
        y_hat=hypothesis(X[i],theta)
        e=e+(y[i]-y_hat)**2

    return e/(2* m)


def gradient(X,y,theta):
    m=X.shape[0]
    grad=np.zeros((2,))
    for i in range(m):
        exp=hypothesis(X[i],theta)-y[i]
        grad[0] += exp
        grad[1]+= exp * X[i]
    return grad/m

def train(X,y,learning_rate=0.5):
    theta=np.zeros((2,))
    error_list=[]
    maxtrs=100
    for i in range(maxtrs):
        grad=gradient(X,y,theta)
        error_list.append(error(X,y,theta))
        theta[0]=theta[0]-learning_rate*grad[0]
        theta[1]=theta[1]-learning_rate*grad[1]
    plt.plot(error_list)
    plt.show()
    return theta


theta=train(X,y)
        



