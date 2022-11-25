import numpy as np
import pandas as pd
import random

def iris_type(s):
    it = {b'Iris-setosa':0, b"Iris-versicolor":1, b'Iris-virginica':2}
    return it[s]

def train_test_split(x, y, rate):
    sample=random.sample(range(len(x)), int(rate*len(x)))
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    for i in range(len(x)):
        if (i in sample):
            x_train.append(x[i, :])
            y_train.append(x[i, :])
        else:
            x_test.append(x[i, :])
            y_test.append(x[i, :])
    return np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)

def train_test_split2(x, y):
    length=x.shape[0]
    index=np.random.randint(0, length+1,size=[length])
    index=set(index)
    x_train=[]
    x_test=[]
    y_train = []
    y_test = []

    for i in range(length):
        if(i in index):
            x_train.append(x[i,:])
            y_train.append(y[i, :])
    return np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)

def train_test_split3(x, y):
    sample=random.sample(range(len(x)),len(x))
    x2=[]
    y2=[]
    for idx in sample:
        x2.append(x[idx, :])
        y2.append(y[idx, :])
    length=x.shape[0]
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]
    index=[i*length//10 for i in range(10)]
    index.append(length)
    x_temp=[]
    y_temp=[]
    for i in range(10):
        x_temp.append(x2[index[i]:index[i+1]])
        y_temp.append(y2[index[i]:index[i + 1]])

    for i in range(10):
        x_train_temp=[]
        x_test_temp=[]
        y_train_temp = []
        y_test_temp = []
        for j in range(10):
            if(i !=j):
                if(len(x_train_temp)==0):
                    x_train_temp=x_temp[j]
                    y_train_temp = y_temp[j]
                else:
                    x_train_temp=np.concatenate([x_train_temp, x_temp[j]], axis=0)
                    y_train_temp = np.concatenate([y_train_temp, y_temp[j]], axis=0)
            else:
                x_test_temp=x_temp[j]
                y_test_temp = y_temp[j]
        x_train.append(x_train_temp)
        x_test.append(x_test_temp)
        y_train.append(y_train_temp)
        y_test.append(y_test_temp)
    return np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)


def testModel(x_train, y_train, x_test, y_test, model):
    H=model.fit(x_train,y_train)
    y_hat=model.predict(x_test)
    labels=np.argmax(y_test, axis=1)
    acc=(y_hat==labels).sum()/len(y_hat)
    err=1-acc
    TP=np.logical_and((y_hat==0), (labels==0)).sum()
    FP = np.logical_and((y_hat == 0), (labels != 0)).sum()
    FN = np.logical_and((y_hat != 0), (labels == 0)).sum()
    TN = np.logical_and((y_hat != 0), (labels != 0)).sum()
    P=TP/(TP+FP)
    R=TP/(TP+FN)
    F1=2*P*R/(P+R)
    return err, acc, P, R, F1, model, H

def testModel2(x_train, y_train, x_test, y_test, model):
    err=acc=P=R=F1=0
    for i in range(len(x_train)):
        err_tmp, acc_tmp, P_tmp, R_tmp, F1_tmp, _, _=testModel(x_train[i], y_train[i], x_test[i], y_test[i], model)
        err+=err_tmp
        acc+=acc_tmp
        P+=P_tmp
        R+=R_tmp
        F1+=F1_tmp
    return err/10, acc/10, P/10, R/10, F1/10

def calc_ROC(x_test, y_test, model):
    label=np.argmax(y_test, axis=1)
    pos=(label==0).sum()
    neg=(label !=0).sum()
    prob=model.predict_prob(x_test)[:,0]
    idx=np.argsort(prob)[-1::-1]
    X=[0]
    Y=[0]

    for i in idx:
        if(label[i]==0):
            X.append(X[-1])
            Y.append(Y[-1]+1/pos)
        else:
            X.append(X[-1]+1/neg)
            Y.append(Y[-1])
    return X, Y

class NN:
    def __init__(self, x_size, hidden_size, y_size):
        self.layers_size=len(hidden_size, y_size)
        self.weights=[]
        self.bias=[]
        self.weights.append(np.random.randn(x_size, hidden_size[0])*0.01)
        self.bias.append(np.ones([1, hidden_size[0]]))
        for i in range(len(hidden_size)-1):
            self.weights.append(np.random.randn(hidden_size[i], hidden_size[i+1])*0.01)
            self.bias.append(np.opnes([i, hidden_size[i+1]])*0.01)
        self.weights.append(np.random.randn(hidden_size[-1], y_size)*0.01)
        self.bias.append(np.ones([1, y_size])*0.01)

    def relu(self, x):
        x[x<0]=0
        return x
    def drelu(self, da, z):
        dz[z<=0]=0
        return dz

    def softmax(self,z):
        y=np.exp(z)
        fm=np.sum(y, axis=1, keepdims=True)
        fm=np.dot(fm,[[1]*z.shape[1]])
        return y/fm

    def dsoftmax(self, da, z):
        dz=np.zeros(z.shape)
        for j in range(da.shape[1]):
            for k in range(da.shape[1]):
                if(j==k):
                    dz[:, j]+=da[:, k]*z[:, j]*(1-z[:, k])
                else:
                    dz[:, j]-=da[:, k]*z[:, j]*z[:,k]
        return dz
    def loss(self, y_hat, y):
        tmp=y*np.log(y_hat)
        L=-np.sum(tmp)/len(y)
        return L
    def dloss(self, y_hat, y):
        return -y/y_hat/len(y)

    def grad(self, x, y):
        temp=np.ones([len(x), 1])
        tempx=x
        dw=[]
        db=[]
        caches=[]
        atcaches=[]
        for i in range(self.layers_size-1):
            x=np.dot(x, self.weights[i])
            x+=np.dot(temp, self.bias[i])
            caches.append(x)
            x=self.relu(x)
            atcaches.append(x)
        z=np.dot(x, self.weights[-1])
        z+=np.dot(temp, self.bias[-1])
        y_hat=self.softmax(z)
        da=self.dloss(y_hat, y)
        dz=self.dsoftmax(da, z)

        for i in range(1, len(caches)+1):
            tmp1=np.dot(atcaches[-i].T, dz)
            tmp2=np.sum(dz, axis=0, keepdims=True)
            dw.insert(0, tmp1/len(x))
            db.insert(0, tmp2/len(x))
            return dw, db

    def fit(self, x, y, lr=0.002, epochs=500, batch_size=30):
        history=[[],[]]
        wv=[]
        wm=[]
        bv=[]
        bm=[]
        for i in range(len(self.weights)):
            wv.append(np.zeros(self.weights[i].shape))
            wm.append(np.zeros(self.weights[i].shape))
            bv.append(np.zeros(self.bias[i].shape))
            bm.append(np.zeros(self.bias[i].shape))
        for i in range(epochs):
            sample=random.sample(range(len(x)),len(x))
            X=x[sample,:]
            Y=y[sample,:]
            for j in range(len(x)//batch_size):
                xx=X[batch_size*j:batch_size*(j+1),:]
                yy=Y[batch_size*j:batch_size*(j+1),:]
                if(j%100==0):
                    y_hat=self.predict_prob(xx)
                    L=self.loss(y_hat, yy)
                    pred=np.argmax(y_hat, axis=1)
                    A=(pred==np.argmax(yy, axis=1)).sum()/len(yy)
                    history[0].append(L)
                    history[1].append(A)
                dw, db=self.grad(xx,yy)
                for k in range(len(dw)):
                    wm[k]=beta1*wm[k]+(1-beta1)*dw[k]
                    wv[k]=beta2*wv[k]+(1-beta2)*dw[k]**2
                    bm[k]=beta1*bm[k]+(1-beta1)*db[k]
                    bv[k]=beta2*bv[k]+(1-beta2)*db[k]**2
                for k in range(len(dw)):
                    wm_hat=wm[k]/(1-beta1**(i+1))
                    wv_hat=wv[k]/(1-beta2**(i+1))
                    bm_hat=bm[k]/(1-beta1**(i+1))
                    bv_hat=bv[k]/(1-beta2**(i+1))
                    self.weights[k]-=lr*wm_hat/(np.sqrt(wv_hat)+epsilon)
                    self.bias[k]-=lr*bm_hat/(np.sqrt(bv_hat)+epsilon)
        return history
    def predict(self, x):
        y_hat=self.predict_prob(x)
        return np.argmax(y_hat, axis=1)
    def predict_prob(self, x):
        temp=np.ones([len(x),1])
        for i in range(self.layers_size-1):
            x=np.dot(x, self.weights[i])
            x+=np.dot(temp, self.bias[i])
            x=self.relu(x)
        x=np.dot(x, self.weights[-1])
        x+=np.dot(temp, self.bias[-1])
        y_hat=self.softmax(x)
        return y_hat

beta1=0.9
beta2=0.999
epsilon=1e-8

data=pd.read_csv('iris.data.csv')

x, y=np.split(data, (4,),axis=1)
mean=np.mean(x, axis=0)
mean=np.dot(np.ones([len(x), 1]), mean)
std=np.std(x, axis=0)
std=np.dot(np.ones([len(x),1]), std)
x=(x-std)/std

tmp=[]
for label in y:
    if(label==0):
        tmp.append([1,0,0])
    elif(label==1):
        tmp.append([0,1,0])
    else:
        tmp.append([0,0,1])
y=np.array(tmp)

x_train, x_test, y_train, y_test=train_test_split(x, y, 0.8)
x_train2, x_test2, y_train2, y_test2=train_test_split2(x, y)
x_train3, x_test3, y_train3, y_test3=train_test_split3(x, y)

model=NN(x.shape[1], [16,8], y.shape[1])
model2=NN(x.shape[1], [16,8], y.shape[1])
model3=NN(x.shape[1], [16,8], y.shape[1])

err, acc, P, R, F1, model4, H=testModel(x_train, y_train, x_test, y_test, model)
err2, acc2, P2, R2, F12, model5, H2=testModel(x_train2, y_train2, x_test2, y_test2, model2)
err3, acc3, P3, R3, F13=testModel2(x_train3, y_train3, x_test3, y_test3, model3)












