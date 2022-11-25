import numpy as np

def f(x, y):
    return (x + y - 3) ** 2 + (x + 2 * y - 5) ** 2 + 2
def f_x_1(x, y):
    return 4 * x + 6 * y - 16
def f_y_1(x, y):
    return 6 * x + 10 * y - 26

learning_rate = 0.1
max_loop = 300
tolerance = 0.0001
x_init = 10
x =x_init

y_init = 10
y =y_init
fx_pre = 0
for i in range(max_loop):
    d_f_x = f_x_1(x,y)
    x=x - learning_rate * d_f_x

    d_f_y = f_y_1(x, y)
    y = y - learning_rate * d_f_y
    #print(x,y)

print('initial x =',x_init)
print('initial y =',y_init)
print('arg min f(x,y) of x = '+str(x)+'y = '+str(y))
print('f(x,y) = ',f(x,y))




print('############### Adam #################')

def adam(x, y, beta_1 = 0.9, beta_2 = 0.99, epsilon = 1e-8, lr = 0.01, max_loop = 300):
    m = 0
    g = 0
    for i in range(max_loop):

        m_x = beta_1 * m + (1-beta_1) * f_x_1(x,y)
        g_x = beta_2 * g + (1-beta_2) * f_x_1(x, y)**2

        m_x_hat = m_x / (1-beta_1**(i+1))
        g_x_hat = g_x / (1-beta_2**(i+1))

        m_y = beta_1 * m + (1 - beta_1) * f_y_1(x,y)
        g_y = beta_2 * g + (1 - beta_2) * f_y_1(x, y)**2

        m_y_hat = m_y / (1 - beta_1 ** (i + 1))
        g_y_hat = g_y / (1 - beta_2 ** (i + 1))

        x = x - lr / (((g_x_hat) + epsilon)) ** 0.5 * m_x_hat
        y = y - lr/((g_y_hat) ** 0.5+epsilon) * m_y_hat

    return f(x, y)

x = 1
y = 1
print('min f(x,y) from Adam = ', adam(x,y))