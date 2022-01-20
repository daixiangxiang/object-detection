# _*_coding:utf-8_*_
# 开发人：戴祥祥
# 开发时间：2021-05-06  13:30
# 文件名：peocess.py
import numpy as np
import math
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

mu = 7
server = 1


def intensity(t):
    return 8.924 - 1.584 * math.cos(math.pi * t / 1.51) + 7.897 * math.sin(math.pi * t / 3.02) - \
           10.3434 * math.cos(math.pi * t / 4.53) + 4.293 * math.cos(math.pi * t / 6.04)


def func(t, y):
    res = []
    intense = intensity(t)
    res.append(-intense * y[0] + mu * y[1])
    for i in range(1, len(y) - 1):
        res.append(intense * y[i - 1] - (intense + server * mu) * y[i] + server * mu * y[i + 1])
    res.append(intense * y[-2] - server * mu * y[-1])
    res = np.array(res)
    return res


if __name__ == '__main__':
    checkpoint = np.linspace(0., 8.)
    inten = []
    for i in checkpoint:
        inten.append(intensity(i))
    plt.subplot(1, 2, 1)
    plt.plot(checkpoint, inten)
    sol = solve_ivp(func, [0, 8], np.array([1, 0, 0, 0, 0, 0]), t_eval=checkpoint)
    plt.subplot(1, 2, 2)
    plt.plot(checkpoint, (sol.y)[5])
    plt.show()
