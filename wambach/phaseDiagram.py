#!python

from math import pi
import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def Epi(k, ux):
    return np.sqrt(k**2 + 2*ux)


def Esig(k, x, ux, uxx):
    return np.sqrt(k**2 + 2*ux + 4*x*uxx)


def Eq(k, g, x):
    return np.sqrt(k**2 + g**2*x)


def flow_eq(u, k, N, g, mu, T, x, dx):

    uxForw = (-3.0/2.0*u[0] + 2.0*u[1] - 1.0/2.0*u[2])/dx
    uxBack = (3.0/2.0*u[N-1] - 2.0*u[N-2] + 1.0/2.0*u[N-3])/dx
    uxCent = (-1.0/2.0*u[0:-2] + 1.0/2.0*u[2:])/dx
    ux = np.append(np.append(uxForw, uxCent), uxBack)
    uxxForw = (2.0*u[0] - 5.0*u[1] + 4.0*u[2] - 1.0*u[3])/dx**2
    uxxBack = (2.0*u[N-1] - 5.0*u[N-2] + 4.0*u[N-3] - 1.0*u[N-4])/dx**2
    uxxCent = (u[0:-2] - 2*u[1:-1] + u[2:])/dx**2
    uxx = np.append(np.append(uxxForw, uxxCent), uxxBack)

    # Compute du/dk.
    dudk = k**4/(12*pi**2)*(3.0/Epi(k, ux)*(1.0/np.tanh(Epi(k, ux)/(2*T)))
                            + 1.0/(Esig(k, x, ux, uxx))
                            * (1.0/np.tanh(Esig(k, x, ux, uxx)/(2*T)))
                            - 12.0/Eq(k, g, x)*(np.tanh((Eq(k, g, x) - mu)/(2*T))
                            + np.tanh((Eq(k, g, x) + mu)/(2*T))))

    # print(k, N, g, mu, T, dx, ux, Epi(k, ux))
    print(k)
    return dudk


def solution(u0, k, N, g, mu, T, x, dx):
    sol = odeint(flow_eq, u0, k, args=(N, g, mu, T, x, dx, ), mxstep = 5000,
                 rtol = 1e-10, atol = 1e-10)
    # print sol
    return sol


if __name__ == "__main__":
    lam = 10.0
    g = 3.2
    k_cutoff = 500
    k_IR = 40
    N_k = 100
    L = 150.0**2
    N = 60
    dx = L / (N - 1.0)
    x = np.linspace(0, L, N)
    k = np.linspace(k_cutoff, k_IR, N_k)
    u0 = lam/4.0*x**2
    T_max = 190
    N_T = 5
    mu_max = 320
    N_mu = 5
    T_array = np.linspace(5, T_max, N_T)
    mu_array = np.linspace(0, mu_max, N_mu)

    min_values = np.empty([len(mu_array), len(T_array)])
    # print(T_array, mu_array)
    for m in range(len(mu_array)):
        for t in range(len(T_array)):
            input((mu_array[m], T_array[t]))
            sol = solution(u0, k, N, g, mu_array[m], T_array[t], x, dx)
            min_values[m, t] = np.sqrt(np.argmin([sol[-1, :]])*dx)

    print(min_values)

    mu_ax, T_ax = np.meshgrid(mu_array, T_array)
    print(mu_ax, T_ax, min_values)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surface = ax.plot_surface(mu_ax, T_ax, min_values.T, rstride=1, cstride=1,
                              cmap='viridis', edgecolor='none')
    ax.set_title('Phase Diagram')
    plt.savefig('wambachPhaseDiagram.png')
    plt.show()
