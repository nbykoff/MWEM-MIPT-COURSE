import numpy as np
from celluloid import Camera
from matplotlib import pyplot as plt

r_min = 0.0
r_max = 1.8
c = 1.5
a = 0.6
b = 1.2

N = 200
dim = 1
C = 0.4
M = 1.5
k = 3

h = (r_max - r_min) / N
tau = C * h / c
nt = int(M / tau)
rs = np.arange(r_min - h / 2, r_max + h, h)


def v0(x):
    if a < x < b:
        return np.exp((-4 * (2 * x - (a + b)) * (2 * x - (a + b))) /
                      ((b - a) * (b - a) - (2 * x - (a + b)) * (2 * x - (a + b))))
    else:
        return 0


def dv(x):
    if a < x < b:
        return - ((a - b) ** 2 * (-a - b + 2 * x) * np.exp(-((a + b - 2 * x) ** 2 / ((a - x) * (x - b))))) / (
                (a - x) ** 2 * (b - x) ** 2)
    else:
        return 0


def analit_u(r, x):
    return r ** ((1 - dim) / 2) * v0(x)


def analit_du(r, x):
    return ((1 - dim) / 2) * r ** (-(1 + dim) / 2) * v0(x) - r ** ((1 - dim) / 2) * dv(x)


def u(N):
    h = (r_max - r_min) / N
    tau = C * h / c
    nt = int(M / tau)
    rs = np.arange(r_min - h / 2, r_max + h, h)

    # coefs
    A = [tau * tau * c * c * rs[i] ** (1 - dim) / h * (rs[i] + h / 2) ** (dim - 1) / h for i in range(N + 2)]
    B = [tau * tau * c * c * rs[i] ** (1 - dim) / h * (rs[i] - h / 2) ** (dim - 1) / h for i in range(N + 2)]

    u0 = [v0(rs[i]) for i in range(N + 2)]
    u1 = [0] * (N + 2)
    u_der2 = [0] * (N + 2)
    u_n = [0] * (N + 2)

    for i in range(1, N + 1):
        u_der2[i] = A[i] * (u0[i + 1] - u0[i]) - B[i] * (u0[i] - u0[i - 1])
        u1[i] = u0[i] + 0.5 * u_der2[i]
    u1[N + 1] = u0[N + 1] + 0.5 * u_der2[N + 1]

    res = [[0 for _ in range(N)] for _ in range(nt)]
    res[0] = u0[1:N + 1]
    res[1] = u1[1:N + 1]

    for i in range(1, nt):
        for j in range(1, N + 1):
            u_n[j] = 2 * u1[j] - u0[j] + A[j] * (u1[j + 1] - u1[j]) - B[j] * (u1[j] - u1[j - 1])
        u_n[0] = u_n[1]
        u_n[-1] = u_n[-2]
        u0 = u1.copy()
        u1 = u_n.copy()
        res[i] = u0[1:N + 1]
    return res


def analytic(N):
    h = (r_max - r_min) / N
    tau = C * h / c
    nt = int(M / tau)
    rs = np.arange(r_min - h / 2, r_max + h, h)
    res = np.zeros((nt, N + 2))
    for n in range(nt):
        for i in range(N + 2):
            res[n][i] = rs[i] ** ((1 - dim) / 2) * v0(-c * tau * n + rs[i])
    return res


def animate(N, nt):
    X = np.linspace(0, 1.8, N)
    fig = plt.figure()
    camera = Camera(fig)
    plt.grid("ON")
    # plt.xlim(0.0, 1.75)
    # plt.ylim(0.0, 1.0)
    for i in range(0, nt):
        b = u1[i][1:N + 1]
        plt.plot(X, u1[i])
        plt.plot(X, b)
        plt.text(0.1, 1.0, "{:f}".format(i * tau))
        camera.snap()
        animation = camera.animate()
    plt.show()


def l2_norma(u1, u2, u3):
    norma = [0] * nt
    for i in range(nt):
        top = 0
        bot = 0
        for j in range(N):
            top += abs(u1[i][j] - u2[3 * i][3 * j + 1]) ** 2
            bot += abs(u2[3 * i][3 * j + 1] - u3[9 * i][9 * j + 4]) ** 2
        norma[i] = np.sqrt(top / bot)
    return norma


def c_norma(u1, u2, u3):
    norma = [0.] * nt
    top = [[0 for _ in range(N)] for _ in range(nt)]
    bot = [[0 for _ in range(N)] for _ in range(nt)]
    for i in range(nt):
        for j in range(N):
            top[i][j] = u1[i][j] - u2[3 * i][3 * j + 1]
            bot[i][j] = u2[3 * i][3 * j + 1] - u3[9 * i][9 * j + 4]
        norma[i] = max(top[i]) / max(bot[i])
    return norma


u1 = u(N)
u2 = u(k * N)
u3 = u(k * k * N)
time = [i * tau for i in range(nt)]
l2 = l2_norma(u1, u2, u3)
Cn = c_norma(u1, u2, u3)
# X = np.linspace(0, 1.8, N)
plt.plot(time, l2, label='L2-norm')
plt.plot(time, Cn, label='C-norm')
plt.plot(time, [9 for _ in time], label='Ref=9')
plt.title(u"O2 for 1D in R{} (grid convergence)".format(dim))
plt.xlim(0.0, 1.5)
plt.grid()
plt.legend()
plt.show()
# animate(N, nt)
