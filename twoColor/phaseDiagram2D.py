#!python

from math import pi
import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt


def Epi(k, ux):
    return np.sqrt((k**2 + 2*ux)*np.heaviside(k**2 + 2*ux, 0)) \
        + 1.*(1 - np.heaviside(k**2 + 2*ux, 0))


def Esig(k, ux, uxx, xx):
    return np.sqrt((k**2 + 2*ux + 4*xx*uxx)*np.heaviside(k**2 + 2*ux + 4*xx
                                                         * uxx, 0))\
        + 1.*(1 - np.heaviside(k**2 + 2*ux + 4*xx*uxx, 0))


"""
def Epi(k, ux):
    return np.sqrt(k**2 + 2*ux)


def Esig(k, gr, ux, uxx):
    return np.sqrt(k**2 + 2*ux + 4*gr*uxx)
"""


def Eq(k, g, xx):
    return np.sqrt(k**2 + g**2*xx)


def flow_eq(u, k, N, Nd, g, mu, T, gr, dx, xx, yy):
    uxForw = (-3.0/2.0*u[0] + 2.0*u[1] - 1.0/2.0*u[2])/dx
    uxBack = (3.0/2.0*u[N-1] - 2.0*u[N-2] + 1.0/2.0*u[N-3])/dx
    uxCent = (-1.0/2.0*u[0:N-2] + 1.0/2.0*u[2:N])/dx
    ux = np.append(np.append(uxForw, uxCent), uxBack)
    for i in range(N, N*Nd, N):
        uxForw = (-3.0/2.0*u[0 + i] + 2.0*u[1 + i] - 1.0/2.0*u[2 + i])/dx
        uxBack = (3.0/2.0*u[i + N-1] - 2.0*u[i + N-2] + 1.0/2.0*u[i + N-3])/dx
        uxCent = (-1.0/2.0*u[i+0:N+i-2] + 1.0/2.0*u[i+2:N+i])/dx
        ux = np.append(ux, np.append(np.append(uxForw, uxCent), uxBack))
    uxxForw = (2.0*u[0] - 5.0*u[1] + 4.0*u[2] - 1.0*u[3])/dx**2
    uxxBack = (2.0*u[N-1] - 5.0*u[N-2] + 4.0*u[N-3] - 1.0*u[N-4])/dx**2
    uxxCent = (u[0:N-2] - 2*u[1:N-1] + u[2:N])/dx**2
    uxx = np.append(np.append(uxxForw, uxxCent), uxxBack)
    for i in range(N, N*Nd, N):
        uxxForw = (2.0*u[i] - 5.0*u[i+1] + 4.0*u[i+2] - 1.0*u[i+3])/dx**2
        uxxBack = (2.0*u[i+N-1] - 5.0*u[i+N-2] + 4.0*u[i+N-3]
                   - 1.0*u[i+N-4])/dx**2
        uxxCent = (u[i+0:i+N-2] - 2*u[i+1:i+N-1] + u[i+2:i+N])/dx**2
        uxx = np.append(uxx, np.append(np.append(uxxForw, uxxCent), uxxBack))

    dudk = k**4/(12*pi**2)*(3.0/Epi(k, ux)*(1.0/np.tanh(Epi(k, ux)/(2*T)))
                            + 1.0/(Esig(k, ux, uxx, xx))
                            * (1.0/np.tanh(Esig(k, ux, uxx, xx)/(2*T)))
                            - 8.0/Eq(k, g, xx)*(np.tanh((Eq(k, g, xx) - mu)
                                                        / (2*T))
                            + np.tanh((Eq(k, g, xx) + mu)/(2*T))))
    # print(k, N, g, mu, T, dx, ux, Epi(k, ux))
    # print(k)
    return dudk


def solution(u0, k, N, Nd, g, mu, T, gr, dx, xx, yy):
    sol = odeint(flow_eq, u0, k, args=(N, Nd, g, mu, T, gr, dx, xx, yy),
                 mxstep=1000, rtol=1e-9)
    return sol


if __name__ == "__main__":
    lam = 23
    m_lam = 0
    g = 4.8
    k_cutoff = 600
    k_IR = 100
    N_k = 50
    L = 150.0**2
    N = 60
    Nd = 2
    dx = L / (N)
    dy = L / (Nd)
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, Nd)
    gr = np.linspace(0, L, N)
    xx = x
    yy = y
    for i in range(Nd - 1):
        # create rho vector for functions that only depend on rho
        xx = np.append(xx, x)
    for i in range(N - 1):
        # create Delta vector for functions that only depend on Delta
        yy = np.append(yy, y)
    for i in range(Nd - 1):
        gr = np.append(gr, np.linspace(y[i + 1], L + y[i + 1], N))
    k = np.linspace(k_cutoff, k_IR, N_k)
    u0 = 1.0/2.0*m_lam**2*gr + lam/4.0*gr**2
    T_min = 3
    T_max = 220
    N_T = 6
    mu_max = 420
    N_mu = 6
    T_array = np.linspace(T_min, T_max, N_T)
    mu_array = np.linspace(0, mu_max, N_mu)
    expl_sym_br = 1450000*np.sqrt(xx)
    complete_sol = [[None for _ in range(len(mu_array))] for _ in
                    range(len(T_array))]
    min_values = np.empty([len(mu_array), len(T_array)])

    for m in range(len(mu_array)):
        for t in range(len(T_array)):
            print((mu_array[m], T_array[t]))
            sol = solution(u0, k, N, Nd, g, mu_array[m], T_array[t], gr, dx,
                           xx, yy)
            complete_sol[m][t] = sol
            min_values[m, t] = np.sqrt(np.argmin([sol[-1, :] - expl_sym_br])
                                       * dx)

    # print("expl", expl_sym_br)
    # print("complete", (complete_sol[0][0] - expl_sym_br)[-1, :])
    print("chiral condensate: "+str(min_values[0, 0]))
    """
    param_list = np.array([lam, m_lam, g, k_cutoff, k_IR, N_k, L, N, dx, T_min,
                           T_max, N_T, mu_max, N_mu])
    dat_name = 'StrodthoffPhaseDiagramN_T'+str(N_T)+'N_mu'+str(N_mu)
    params_name = dat_name+'params'
    fig_name = 'StrodthoffPhaseDiagramN_T'+str(N_T)+'N_mu'+str(N_mu)+'.png'
    np.save(dat_name, complete_sol)
    np.save(params_name, param_list)
    """
    mu_ax, T_ax = np.meshgrid(mu_array, T_array)
    fig = plt.figure()
    CS = plt.contourf(mu_ax, T_ax, min_values.T, 16)
    plt.title('Phase Diagram')
    # plt.savefig(fig_name)
    plt.show()
