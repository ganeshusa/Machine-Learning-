import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


#Data Generate X,Y
def generateDataSet(m):
    X = np.random.randn(m) * 10
    noise = np.random.randn(m)
    y = 3 * X + 1 + 5 * noise
    return X, y


X, y = generateDataSet(100)
print(X.shape, y.shape)


def plotData(X, y, colr="orange", title="Data"):
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(X, y)
    plt.scatter(X, y, color=colr)
    plt.show()


def normalizeData(X):
    X = (X - X.mean()) / X.std()
    return X


plotData(X, y)
X = normalizeData(X)
plotData(X, y)


def trainTestSplit(X, y, split=0.8):
    m = X.shape[0]
    data = np.zeros((m, 2))
    data[:, 0] = X
    data[:, 1] = y
    np.random.shuffle(data)
    split = int(m * split)

    XT = data[:split, 0]
    yT = data[:split, 1]

    xt = data[split:, 0]
    yt = data[split:, 1]
    return XT, yT, xt, yt


XT, yT, Xt, yt = trainTestSplit(X, y)
print(XT.shape, yT.shape)
print(Xt.shape, yT.shape)

plt.scatter(XT, yT, color="orange", label="training data")
plt.scatter(Xt, yt, color="blue", label="test data")
plt.title("Training and test Data")
plt.legend()
plt.show()


def hypothesis(X, theta):
    return theta[0] + theta[1] * X


def error(X, y, theta):
    m = X.shape[0]
    e = 0
    for i in range(m):
        y_hat = hypothesis(X[i], theta)
        e = e + (y[i] - y_hat) ** 2

    return e / (2 * m)


def gradient(X, y, theta):
    m = X.shape[0]
    grad = np.zeros((2,))
    for i in range(m):
        exp = hypothesis(X[i], theta) - y[i]
        grad[0] += exp
        grad[1] += exp * X[i]
    return grad / m


def train(X, y, learning_rate=0.1):
    theta = np.zeros((2,))
    error_list = []
    maxtrs = 100
    for i in range(maxtrs):
        grad = gradient(X, y, theta)
        error_list.append(error(X, y, theta))
        theta[0] = theta[0] - learning_rate * grad[0]
        theta[1] = theta[1] - learning_rate * grad[1]
    plt.plot(error_list)
    plt.show()
    return theta


theta = train(X, y)
print(theta)


def predict(X, theta):
    return hypothesis(X, theta)


yp = predict(Xt, theta)
plt.scatter(XT, yT, label="training data")
plt.scatter(Xt, yt, color="orange", label="test data")
plt.plot(Xt, yp, color="blue", label="prediction data")
plt.legend()
plt.show()


def r2_score(y, yp):
    ymean = y.mean()
    num = np.sum((y - yp) ** 2)
    denom = np.sum((y - ymean) ** 2)
    return 1 - num / denom

r2 = r2_score(yt, yp)
print(r2)
