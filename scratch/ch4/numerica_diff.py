import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x = x - lr * grad
    return x

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

def function_2(x):
    return x[0] ** 2 + x[1] ** 2

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))

init_x = np.array([-3.0, 4.0])
t = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(t)



'''
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)

k1 = numerical_diff(function_1, 5)
y1 = k1 * x + (function_1(5) - k1 * 5)
plt.plot(x, y1)

k2 = numerical_diff(function_1, 15)
y2 = k2 * x + (function_1(15) - k2 * 15)
plt.plot(x, y2)

plt.show()
'''
