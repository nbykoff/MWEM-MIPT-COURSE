import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera


# задаём размерность R1(R2, R3)
dim = 3

r_min = 0
r_max = 1.8
c = 1.5
a = 0.6
b = 1.2

I = 200
C = 0.95
M = 3.

h = (r_max - r_min) / I
tau = C * h / c
nt = int(M / tau)
rs = np.arange(r_min - h / 2, r_max + h, h)

def v0(x):
    if a < x < b:
        return np.exp((-4 * (2 * x - (a + b)) * (2 * x - (a + b))) /
                      ((b - a) * (b - a) - (2 * x - (a + b)) * (2 * x - (a + b))))
    else:
        return 0

def u(I):
    # coefs
    A = [tau * tau * c * c * rs[i] ** (1 - dim) / h * (rs[i] + h / 2) ** (dim - 1) / h for i in range(I + 2)]
    B = [tau * tau * c * c * rs[i] ** (1 - dim) / h * (rs[i] - h / 2) ** (dim - 1) / h for i in range(I + 2)]

    u0 = [v0(rs[i]) for i in range(I + 2)]
    u1 = [0] * (I + 2)
    u_der2 = [0] * (I + 2)
    u_n = [0] * (I + 2)

    for i in range(1, I + 1):
        u_der2[i] = A[i] * (u0[i + 1] - u0[i]) - B[i] * (u0[i] - u0[i - 1])
        u1[i] = u0[i] + 0.5 * u_der2[i]
    u1[I + 1] = u0[I + 1] + 0.5 * u_der2[I + 1]

    res = [[0 for _ in range(I)] for _ in range(nt)]
    res[0] = u0[1:I + 1]
    res[1] = u1[1:I + 1]

    for i in range(1, nt):
        for j in range(1, I + 1):
            u_n[j] = 2 * u1[j] - u0[j] + A[j] * (u1[j + 1] - u1[j]) - B[j] * (u1[j] - u1[j - 1])
        u_n[0] = u_n[1]
        u_n[-1] = u_n[-2]
        u0 = u1.copy()
        u1 = u_n.copy()
        res[i] = u0[1:I + 1]
    return res

def animate(N):
    X = np.linspace(0, 1.8, N)
    fig = plt.figure()
    camera = Camera(fig)
    plt.grid("ON")
    plt.xlim(0.0, 1.75)
    plt.ylim(0.0, 1.0)
    for i in range(1, nt):
        plt.plot(X, u1[i])
        plt.text(0.1, 1.0, "{:f}".format(i * tau))
        camera.snap()
        animation = camera.animate()
        animation.save('R3.gif', writer = 'imagemagick')
    plt.show()


u1 = u(I)
animate(I)