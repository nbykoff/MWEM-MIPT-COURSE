from math import exp, sqrt

import matplotlib.pyplot as plt
import numpy as np

r_min = 0
r_max = 1.8
c = 1.5
a = -0.7
b = -0.1

N = 100
nt = 70
C = 0.8
dim = 1
k = 3
coefs = [-5. / 4, 4. / 3, -1. / 12]


def v0(r, t):
    x = c * t - r
    if a < x < b:
        return exp((- 4 * (2 * x - (a + b)) ** 2) / ((b - a) ** 2 - (2 * x - (a + b)) ** 2))
    else:
        return 0


def u(I, T, K):
    u = np.zeros((I + 2, T))
    h = (r_max - r_min) / I
    tau = C * h / c / K
    rs = [r_min + (i - 0.5) * h for i in range(I + 2)]

    for i in range(1, I + 1):
        u[i, 0] = v0(rs[i], 0)

    for i in range(1, I + 1):
        u[i, 1] = v0(rs[i], tau)

    for n in range(1, T - 1):
        for i in range(2, I):
            u[i, n + 1] = u_n1(i, n, u, tau, h)
        u[1, n + 1] = u[2, n + 1]
        u[0, n + 1] = u[2, n + 1]
        u[I, n + 1] = u[I - 1, n + 1]
        u[I + 1, n + 1] = u[I - 1, n + 1]
    return u


def u_n1(i, n, u, tau, h):
    sum = 0
    for k in range(0, 3):
        sum += tau ** 2 * c ** 2 / h ** 2 * (u[i - k, n] + u[i + k, n]) * coefs[k]
    return 2 * u[i, n] - u[i, n - 1] + sum


def norm_C(l):
    return np.max(l)


def norm_L2(l):
    return sqrt(np.sum([abs(x) ** 2 for x in l]))


h = (r_max - r_min) / N
tau = C * h / c

u1 = u(N, nt, 1)
u2 = u(N * k, nt * k ** 2, 3)
u3 = u(N * k ** 2, nt * k ** 4, 9)

c_norma = []
for n in range(nt):
    norm1 = norm_C(u3[5::k ** 2, n * k ** 4] - u2[2::k, n * k ** 2])
    norm2 = norm_C(u2[2::k, n * k ** 2] - u1[1:N + 1, n])
    c_norma.append(norm2 / norm1)

l_norma = []
for n in range(nt):
    norm1 = norm_L2(u3[5::k ** 2, n * k ** 4] - u2[2::k, n * k ** 2])
    norm2 = norm_L2(u2[2::k, n * k ** 2] - u1[1:N + 1, n])
    if norm1 != 0:
        l_norma.append(norm2 / norm1)
    else:
        l_norma.append(0)

time = [tau * n for n in range(nt)]
plt.plot(time, l_norma, label='L2')
plt.plot(time, c_norma, label='C')
plt.plot(time, [81] * nt, label='Ref=9')
plt.grid()
plt.legend()
plt.show()
