import numpy as np
from lu import *

a = 1.7
b = 3.2
alpha = 0
beta = 1/4
max_derivative = 129.429
rating_above = 3.9901
real_ans = 11.83933565874812191864851199716726555747 # 2

def p(x):
    return ((x - a) ** (-alpha)) * ((b - x) ** (-beta))

def f(x):
    return 3 * np.cos(0.5 * x) * np.exp(x / 4) + 5 * np.sin(2.5 * x) * np.exp( - x / 3) + 2 * x

def IQF(a , b):
    x_1 = a
    x_2 = (a + b) / 2
    x_3 =b

    x = np.array([x_1, x_2, x_3])

    m_0 = (-4/3 * (16/5 - b)**(3/4)) - (-4/3 * (16/5 - a)**(3/4))
    m_1 = (-4/105 * (16/5 - b)**(3/4) * (64 + 15*b)) - (-4/105 * (16/5 - a)**(3/4) * (64 + 15*a))
    m_2 = (-(4*(16/5-b)**(3/4)*(8192 + 1920*b + 525*b**2))/5775)  - (-(4*(16/5-a)**(3/4)*(8192 + 1920*a + 525*a**2))/5775)
    mu = np.array([m_0, m_1, m_2])

    powers = np.arange(3)[:, None]

    A = x ** powers
    L, U, m_row, m_col = meow(A)
    mu_copy = np.transpose(m_row.dot(mu))

    y_solv = np.zeros_like(mu_copy)
    for i in range(len(mu_copy)):
        y_solv[i] = mu_copy[i] - np.dot(L[i, :i], y_solv[:i])
    dx = np.zeros_like(y_solv)
    for i in range(len(y_solv) - 1, -1, -1):  # идем с конца, реверсом до 0 элемента
        dx[i] = (y_solv[i] - np.dot(U[i, i + 1:], dx[i + 1:])) / U[i, i]
    dx = np.matmul(m_col, dx)

    A_1, A_2, A_3 = dx

    IQF = A_1 * f(x_1) + A_2 * f(x_2) + A_3 * f(x_3)
    return IQF




