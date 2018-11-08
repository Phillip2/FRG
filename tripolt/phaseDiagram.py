#!python

from math import pi
import numpy as np
from scipy.integrate import ode
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


def f(t, y, N, g, mu, T, x, dx):
    u = y
    k = t
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


def ode_solve(N, g, m, t, x, dx, mu_ax, T_ax):
    ode15s.set_initial_value(u0, k_cutoff).set_f_params(N, g, mu_ax[t][m],
                                                        T_ax[t][m], x, dx)
    c = 1
    while c < N_k:
        ode15s.integrate(ode15s.t-dk)
        print(ode15s.t)
        c += 1
    return (ode15s.y, ode15s.t, ode15s.successful())


def solve(N, g, m, t, x, dx, mu_ax, T_ax):
    res = ode_solve(N, g, m, t, x, dx, mu_ax, T_ax)
    print(res[1], res[2])
    print((mu_ax[t][m], T_ax[t][m]))
    return res[0]


def interpFunc(t, a, b, c):
    return a*t**2 + b*t + c


def interpolate(s, expl_sym_br, dx):
    # Quadratische Interpolation vielleicht zu ungenau...
    x1 = np.argmin([s - expl_sym_br])
    x2 = x1 + 1
    x3 = x1 - 1
    y1 = s[x1] - expl_sym_br[x1]
    y2 = s[x2] - expl_sym_br[x2]
    y3 = s[x3] - expl_sym_br[x3]
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
    dk = 1
    N_k = (k_cutoff - k_IR)/(dk) + 1
    L = 140.0**2
    N = 50
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    k = np.linspace(k_cutoff, k_IR, N_k)
    u0 = 1.0/2.0*m_lam**2*x + lam/4.0*x**2
    T_min = 5
    T_max = 200
    N_T = 5
    mu_min = 0
    mu_max = 350
    N_mu = 5
    T_array = np.linspace(T_min, T_max, N_T)
    mu_array = np.linspace(mu_min, mu_max, N_mu)
    mu_ax, T_ax = np.meshgrid(mu_array, T_array)
    h = 1750000
    expl_sym_br = h*np.sqrt(x)
    sol = [[None for _ in range(N_mu)] for _ in
           range(N_T)]
    min_values = np.zeros([N_T, N_mu])
    m_sig = np.zeros([N_T, N_mu])
    m_pi = np.zeros([N_T, N_mu])

    ode_order = 5
    ode_steps = 200000
    ode_integrator = 'vode'
    ode_method = 'bdf'
    r_tol = 1e-13
    a_tol = 1e-13
    ode15s = ode(f)
    ode15s.set_integrator(ode_integrator, method=ode_method, order=ode_order,
                          nsteps=ode_steps, rtol=r_tol, atol=a_tol)

    num_cores = multiprocessing.cpu_count()

    for t in range(N_T):
        sol[t] = Parallel(n_jobs=num_cores)(delayed(solve)(N, g, m, t, x,
                                                              dx, mu_ax, T_ax)
                                            for m in range(N_mu))
    for t in range(N_T):
        for m in range(N_mu):
            s = sol[t][m]
            argmin = np.argmin([s - expl_sym_br])
            if argmin != 0 and argmin != N - 1:
                min_values[t, m] = interpolate(s, expl_sym_br, dx)[0]
                m_pi[t, m] = np.sqrt(h/min_values[t, m])
                m_sig[t, m] = np.sqrt(4*min_values[t, m]**2 *
                                      interpolate(s, expl_sym_br, dx)[2]/dx**2
                                      + m_pi[t, m]**2)
            else:
                min_values[t, m] = np.sqrt(np.argmin([s
                                                      - expl_sym_br])*dx)

    plt.plot(sol[0][0] - expl_sym_br, color="r")
    plt.plot(sol[0][1] - expl_sym_br, color="r")
    plt.plot(sol[0][2] - expl_sym_br, color="r")
    plt.plot(sol[0][3] - expl_sym_br, color="r")
    plt.plot(sol[0][-3] - expl_sym_br, color="b")
    plt.plot(sol[0][-2] - expl_sym_br, color="b")
    plt.plot(sol[0][-1] - expl_sym_br, color="b")
    plt.show()
    print("chiral condensate: "+str(min_values[0, 0]))
    print("vacuum pion mass: "+str(m_pi[0, 0]))
    print("vacuum sigma mass: "+str(m_sig[0, 0]))
    param_list = np.array([lam, m_lam, g, k_cutoff, k_IR, N_k, L, N, T_min, T_max, N_T,
                  mu_min, mu_max, N_mu, h, sol], dtype=object)
    dat_name = 'TripoltPhaseDiagramN_T'+str(N_T)+'N_mu'+str(N_mu)
    fig_name = 'TripoltPhaseDiagramN_T'+str(N_T)+'N_mu'+str(N_mu)+'.png'
    np.save(dat_name, param_list)
    fig = plt.figure()
    CS = plt.contourf(mu_ax, T_ax, min_values, 15)
    plt.title('Phase Diagram')
    plt.savefig(fig_name)
    plt.show()
