import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import cv2
#plt.style.use('ggplot')
X, y = make_blobs(n_samples=2000, n_features=2, centers=3, cluster_std=3, random_state=42)
n_features = X.shape[1]
m = X.shape[0]
print(X.shape, y.shape)
xt = np.array([-10, 5])
for i in range(m):
    if y[i] == 0:
        plt.scatter(X[i, 0], X[i, 1], c='r', label="red")
    elif y[i] == 1:
        plt.scatter(X[i, 0], X[i, 1], c='g', label="blue")
    else:
        plt.scatter(X[i, 0], X[i, 1], c='b', label="yellow")

plt.scatter(xt[0], xt[1], color='orange', marker='*')
plt.show()


def dist(p, q):
    return np.sqrt(np.sum((p - q) ** 2))


def knn(X, y, xt, k=5):
    m = X.shape[0]
    dilst = []
    for i in range(m):
        d = dist(X[i], xt)
        dilst.append((d, y[i]))

    dilst = sorted(dilst)

    dlist = np.array(dilst[:k])
    labels = dlist[:, 1]
    labels, cnts = np.unique(labels, return_counts=True)
    idx = cnts.argmax()
    pred = labels[idx]
    return int(pred)


pred1 = knn(X, y, xt)
print(pred1)
