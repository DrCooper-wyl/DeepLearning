import numpy as np

np.random.seed(1)
x = np.arange(-100,100,2)
epsilon = np.random.normal(size=100)
y = 3 * x + 4 + epsilon

lr = 0.000001
n = 10000
np.random.seed(456)
a = np.random.randint(0,9)
b = np.random.randint(0,9)
loss_record = []


for iter in range(n):
    loss = np.sum(np.power(y-(a*x+b),2))/2
    loss_record.append(loss)

    delta_a = np.sum(x * (a * x + b - y))
    delta_b = np.sum(a * x + b - y)
    a-= lr * delta_a
    b-= lr * delta_b

    if(iter % 2 == 0 or iter == 29):
        print("iter: %3d; loss: %0.3f" % (iter,loss))

def loss(a,b):
    y_0 = a * x + b
    return sum(1/2*1/n*(y-y_0)*(y-y_0))

print('argmin loss of a: ', a)
print('argmin loss of b: ', b)
print("loss最小值",loss(a,b))