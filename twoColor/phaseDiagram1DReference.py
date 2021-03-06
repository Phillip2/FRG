#!python

from math import pi
import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt


def Epi(k, ux):
    return np.sqrt((k**2 + 2*ux)*np.heaviside(k**2 + 2*ux, 0)) \
        + 1.*(1 - np.heaviside(k**2 + 2*ux, 0))


def Esig(k, x, ux, uxx):
    return np.sqrt((k**2 + 2*ux + 4*x*uxx)*np.heaviside(k**2 + 2*ux + 4*x
                                                        * uxx, 0))\
        + 1.*(1 - np.heaviside(k**2 + 2*ux + 4*x*uxx, 0))


"""
def Epi(k, ux):
    return np.sqrt(k**2 + 2*ux)


def Esig(k, x, ux, uxx):
    return np.sqrt(k**2 + 2*ux + 4*x*uxx)
"""


def Eq(k, g, x):
    return np.sqrt(k**2 + g**2*x)


def flow_eq(u, k, N, g, mu, T, x, dx):
    uxForw = (-3.0/2.0*u[0] + 2.0*u[1] - 1.0/2.0*u[2])/dx
    uxBack = (3.0/2.0*u[N-1] - 2.0*u[N-2] + 1.0/2.0*u[N-3])/dx
    uxCent = (-1.0/2.0*u[0:N-2] + 1.0/2.0*u[2:N])/dx
    ux = np.append(np.append(uxForw, uxCent), uxBack)
    uxxForw = (2.0*u[0] - 5.0*u[1] + 4.0*u[2] - 1.0*u[3])/dx**2
    uxxBack = (2.0*u[N-1] - 5.0*u[N-2] + 4.0*u[N-3] - 1.0*u[N-4])/dx**2
    uxxCent = (u[0:N-2] - 2*u[1:N-1] + u[2:N])/dx**2
    uxx = np.append(np.append(uxxForw, uxxCent), uxxBack)

    # Compute du/dk.
    dudk = k**4/(12*pi**2)*(3.0/Epi(k, ux)*(1.0/np.tanh(Epi(k, ux)/(2*T)))
                            + 1.0/(Esig(k, x, ux, uxx))
                            * (1.0/np.tanh(Esig(k, x, ux, uxx)/(2*T)))
                            - 8.0/Eq(k, g, x)*(np.tanh((Eq(k, g, x) - mu)
                                                       / (2*T))
                            + np.tanh((Eq(k, g, x) + mu)/(2*T))))
    input(ux)
    # print(k, N, g, mu, T, dx, ux, Epi(k, ux))
    # print(k)
    return dudk


def solution(u0, k, N, g, mu, T, x, dx):
    sol = odeint(flow_eq, u0, k, args=(N, g, mu, T, x, dx, ), mxstep=1000,
                 rtol=1e-10)
    input(sol[-1, :])
    return sol


if __name__ == "__main__":
    lam = 23
    m_lam = 0
    g = 4.8
    k_cutoff = 600
    k_IR = 100
    N_k = 100
    L = 150.0**2
    N = 8
    dx = L/N
    x = np.linspace(0, L, N)
    print(x)
    k = np.linspace(k_cutoff, k_IR, N_k)
    u0 = 1.0/2.0*m_lam**2*x + lam/4.0*x**2
    T_min = 3
    T_max = 220
    N_T = 3
    mu_max = 420
    N_mu = 3
    T_array = np.linspace(T_min, T_max, N_T)
    mu_array = np.linspace(0, mu_max, N_mu)
    expl_sym_br = 1450000*np.sqrt(x)
    complete_sol = [[None for _ in range(len(mu_array))] for _ in
                    range(len(T_array))]
    min_values = np.empty([len(mu_array), len(T_array)])

    for m in range(len(mu_array)):
        for t in range(len(T_array)):
            # print((mu_array[m], T_array[t]))
            sol = solution(u0, k, N, g, mu_array[m], T_array[t], x, dx)
            complete_sol[m][t] = sol
            min_values[m, t] = np.sqrt(np.argmin([sol[-1, :] - expl_sym_br])
                                       * dx)

    print("u0", u0)
    print("expl", expl_sym_br)
    print("complete", (complete_sol[0][0] - expl_sym_br)[-1, :])
    print("chiral condensate: "+str(min_values[0, 0]))
    mu_ax, T_ax = np.meshgrid(mu_array, T_array)
    fig = plt.figure()
    CS = plt.contourf(mu_ax, T_ax, min_values.T, 16)
    plt.title('Phase Diagram')
    plt.show()
