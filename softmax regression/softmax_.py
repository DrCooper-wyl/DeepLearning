import numpy as np
import pandas as pd
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor':1, b'Iris-virginica':2}
    return it[s]

path = '/Users/yilinwang/Desktop/0422/iris.data'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4, iris_type})

x, y = np.split(data, (4,), axis=1)

def softmax(sample, weight):
    sample = sample.T
    a = np.dot(weight.T, sample)
    a_exp = np.exp(a)
    prob = a.exp / a_exp.sum(0, keepdims = True)
    return prob

def d_softmax(prob):
    return np.diag(prob) - np.dot(prob, prob.T)

def d_loss(x_sample, y_sample, weight):
    y_sample = y_sample.T
    y_hat = softmax(x_sample, weight)
    y = (y_sample - y_hat).T
    N = len(x_sample)
    return np.dot(x_sample.T, y), N

def gradient_descent(x_train, y_train):
    lr = 1e-1
    max_loop = 100
    w_row = 4
    w_column = 3
    W = np.zeros((w_row, w_column))

    for i in range(max_loop):
        prob = softmax(x_train, W)
        print('第{}次迭代的分类准确度accuracy:{}'.format(i+1,accuracy(y_train,prob)))
        loss, N = d_loss(x_train, y_train, W)
        W += lr * 1 / N * loss
        print('第{]次迭代的参数矩阵W={}'.format(i+1, W))
    return W

def accuracy(y_train, prob):
    y_train = y_train.T
    for i in range(prob.shape[1]):
        prob[:, i][list(prob[:, i]).index(max(prob[:, i]))] = 1

    pred = (prob == y_train)
    score = np.sum(pred == True) / prob.shape[1]
    return score

def pred_vector(sample, weight):
    sample_df = pd.DataFrame(sample)
    scaler = MinMaxScaler()
    sample_df = scaler.fit_transform(sample_df)

    a = np.dot(weight.T, sample_df)
    a_exp = np.exp(a)
    prob = a_exp / a_exp.sum(0, keepdims=True)

    type_dict = {0:'第一类：Iris-setosa', 1:'第二类：Iris-versicolor', 2:'第三类：Iris-virginica'}
    return print(type_dict[list(prob[:, :]).index(max(prob[:, :]))])


if __name__ == '__main__':
    filename = 'iris.data'

    data_df = pd.DataFrame(data)
    encoder = LabelEncoder()
    type_num = encoder.fit_transform(data_df[4].values)
    type_num_t = np.array([type_num]).T

    enc = OneHotEncoder()
    type_hot = enc.fit_transform(type_num_t)
    type_hot = type_hot.toarray()

    scaler = MinMaxScaler()
    data_sd = scaler.fit_transform(data_df.iloc[:, [n for n in range(0,4)]])

    x_train, x_test, y_train, y_test = train_test_split(data_sd, type_hot, test_size=0.2, random_state=2)

    W = gradient_descent(x_train, y_train)

    print(accuracy(y_test, softmax(x_test, W)))

    sample = [5.1, 3.7, 1.5, 0.4]
    pred_vector(sample, W)

