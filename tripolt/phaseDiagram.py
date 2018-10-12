#!python

from math import pi
import numpy as np
from scipy.integrate import odeint
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
from scipy.optimize import fmin


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
                            - 12.0/Eq(k, g, x)*(np.tanh((Eq(k, g, x) - mu)
                                                        / (2*T))
                            + np.tanh((Eq(k, g, x) + mu)/(2*T))))
    # print(k)
    return dudk


def solution(u0, k, N, g, mu, T, x, dx):
    sol = odeint(flow_eq, u0, k, args=(N, g, mu, T, x, dx, ), mxstep=200000,
                 rtol=1e-13, atol=1e-13, full_output=False)
    print((mu, T))
    return sol


def interpFunc(t, a, b, c):
    return a*t**2 + b*t + c


def interpolate(s, expl_sym_br, dx):
    # Quadratische Interpolation vielleicht zu ungenau...
    x1 = np.argmin([s[-1, :] - expl_sym_br])
    x2 = x1 + 1
    x3 = x1 - 1
    y1 = s[-1, x1] - expl_sym_br[x1]
    y2 = s[-1, x2] - expl_sym_br[x2]
    y3 = s[-1, x3] - expl_sym_br[x3]
    a = (x1*(y3 - y2) + x2*(y1 - y3) + x3*(y2 - y1))/((x1 - x2)*(x1 - x3) *
                                                      (x2 - x3))
    b = (y2 - y1)/(x2 - x1) - a*(x1 + x2)
    c = y1 - a*x1**2 - b*x1
    minimum = np.sqrt(fmin(interpFunc, x1, args=(a, b, c), xtol=0.001,
                           ftol=1e+6)[0]*dx)
    return minimum, b, 2*a


if __name__ == "__main__":
    lam = 2.0
    m_lam = 794.
    g = 3.2
    k_cutoff = 1000
    k_IR = 100
    N_k = 100
    L = 140.0**2
    N = 50
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    k = np.linspace(k_cutoff, k_IR, N_k)
    u0 = 1.0/2.0*m_lam**2*x + lam/4.0*x**2
    T_max = 250
    N_T = 20
    mu_max = 400
    N_mu = 20
    T_min = 5
    mu_min = 0
    T_array = np.linspace(T_min, T_max, N_T)
    mu_array = np.linspace(mu_min, mu_max, N_mu)
    h = 1750000
    expl_sym_br = h*np.sqrt(x)
    sol = [[None for _ in range(len(mu_array))] for _ in
           range(len(T_array))]
    min_values = np.zeros([len(mu_array), len(T_array)])
    m_sig = np.zeros([len(mu_array), len(T_array)])
    m_pi = np.zeros([len(mu_array), len(T_array)])

    num_cores = multiprocessing.cpu_count()

    for t in range(len(T_array)):
        sol[t] = Parallel(n_jobs=num_cores)(delayed(solution)
                                            (u0, k, N, g, mu_array[m],
                                             T_array[t], x, dx)
                                            for m in range(len(mu_array)))
    for t in range(len(T_array)):
        for m in range(len(mu_array)):
            s = sol[t][m]
            argmin = np.argmin([s[-1, :] - expl_sym_br])
            if argmin != 0 and argmin != N - 1:
                min_values[t, m] = interpolate(s, expl_sym_br, dx)[0]
                m_pi[t, m] = np.sqrt(h/min_values[t, m])
                m_sig[t, m] = np.sqrt(4*min_values[t, m]**2 *
                                      interpolate(s, expl_sym_br, dx)[2]/dx**2
                                      + m_pi[t, m]**2)
            else:
                min_values[t, m] = np.sqrt(np.argmin([s[-1, :]
                                                      - expl_sym_br])*dx)
            # print(min_values)

    print("chiral condensate: "+str(min_values[0, 0]))
    print("vacuum pion mass: "+str(m_pi[0, 0]))
    print("vacuum sigma mass: "+str(m_sig[0, 0]))
    param_list = np.array([lam, m_lam, g, k_cutoff, k_IR, N_k, L, N, dx, T_max,
                           N_T, mu_max, N_mu])
    dat_name = 'TripoltPhaseDiagramN_T'+str(N_T)+'N_mu'+str(N_mu)
    params_name = dat_name+'params'
    fig_name = 'TripoltPhaseDiagramN_T'+str(N_T)+'N_mu'+str(N_mu)+'.png'
    np.save(dat_name, sol)
    np.save(params_name, param_list)
    mu_ax, T_ax = np.meshgrid(mu_array, T_array)
    fig = plt.figure()
    CS = plt.contourf(mu_ax, T_ax, min_values, 15)
    plt.title('Phase Diagram')
    plt.savefig(fig_name)
    plt.show()
