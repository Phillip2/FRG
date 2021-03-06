#!python

# http://modelling3e4.connectmv.com/wiki/Software_tutorial/Integration_of_ODEs#Example_with_coupled_ODEs

from math import pi
import numpy as np
from scipy.integrate import ode
# from function import func
from function2 import func2
import time
from joblib import parallel, delayed
import multiprocessing

import matplotlib.pyplot as plt

"""
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


def Esig(k, ux, uxx, xx):
    return np.sqrt(k**2 + 2*ux + 4*xx*uxx)


def Eq(k, g, xx):
    return np.sqrt(k**2 + g**2*xx)


def Ek(k, g, xx, yy, mu, n):
    return np.sqrt(g**2*yy + (np.sqrt(k**2 + g**2*xx) + (-1)**n*mu)**2)


def f(t, y, N, Nd, g, mu, T, gr, dx, xx, yy):
    u = y
    k = t
    uxForw = (-3.0/2.0*u[0] + 2.0*u[1] - 1.0/2.0*u[2])/dx
    uxBack = (3.0/2.0*u[N-1] - 2.0*u[N-2] + 1.0/2.0*u[N-3])/dx
    uxCent = (-1.0/2.0*u[0:N-2] + 1.0/2.0*u[2:N])/dx
    ux = np.append(np.append(uxForw, uxCent), uxBack)
    for i in range(N, N*Nd, N):
        uxForw = (-3.0/2.0*u[i] + 2.0*u[1+i] - 1.0/2.0*u[2+i])/dx
        uxBack = (3.0/2.0*u[i+N-1] - 2.0*u[i+N-2] + 1.0/2.0*u[i+N-3])/dx
        uxCent = (-1.0/2.0*u[i:N+i-2] + 1.0/2.0*u[i+2:N+i])/dx
        ux = np.append(ux, np.append(np.append(uxForw, uxCent), uxBack))
    uxxForw = (2.0*u[0] - 5.0*u[1] + 4.0*u[2] - 1.0*u[3])/dx**2
    uxxBack = (2.0*u[N-1] - 5.0*u[N-2] + 4.0*u[N-3] - 1.0*u[N-4])/dx**2
    uxxCent = (u[0:N-2] - 2*u[1:N-1] + u[2:N])/dx**2
    uxx = np.append(np.append(uxxForw, uxxCent), uxxBack)
    for i in range(N, N*Nd, N):
        uxxForw = (2.0*u[i] - 5.0*u[i+1] + 4.0*u[i+2] - 1.0*u[i+3])/dx**2
        uxxBack = (2.0*u[i+N-1] - 5.0*u[i+N-2] + 4.0*u[i+N-3]
                   - 1.0*u[i+N-4])/dx**2
        uxxCent = (u[i:i+N-2] - 2*u[i+1:i+N-1] + u[i+2:i+N])/dx**2
        uxx = np.append(uxx, np.append(np.append(uxxForw, uxxCent), uxxBack))

    uyForw = (-3.0/2.0*u[0:N] + 2.0*u[N:2*N] - 1.0/2.0*u[2*N:3*N])/dy
    uyBack = (3.0/2.0*u[(Nd-1)*N:Nd*N] - 2.0*u[(Nd-2)*N:(Nd-1)*N]
              + 1.0/2.0*u[(Nd-3)*N:(Nd-2)*N])/dy
    uyCent = (-1.0/2.0*u[0:(Nd-2)*N] + 1.0/2.0*u[2*N:Nd*N])/dy
    uy = np.append(np.append(uyForw, uyCent), uyBack)

    uyyForw = (2.0*u[0:N] - 5.0*u[N:2*N] + 4.0*u[2*N:3*N] - 1.0*u[3*N:4*N]) \
        / dy**2
    uyyBack = (2.0*u[(Nd-1)*N:Nd*N] - 5.0*u[(Nd-2)*N:(Nd-1)*N]
               + 4.0*u[(Nd-3)*N:(Nd-2)*N] - 1.0*u[(Nd-4)*N:(Nd-3)*N])/dy**2
    uyyCent = (u[0:(Nd-2)*N] - 2*u[N:(Nd-1)*N] + u[2*N:Nd*N])/dy**2
    uyy = np.append(np.append(uyyForw, uyyCent), uyyBack)

    # Check: define second derivative in terms of first derivative
    # should approx. coincide with uyyCent
    # uyyCent2 = (-1.0/2.0*uy[0:(Nd-2)*N] + 1.0/2.0*uy[2*N:Nd*N])/dy

    uxyForw = (-3.0/2.0*ux[0:N] + 2.0*ux[N:2*N] - 1.0/2.0*ux[2*N:3*N])/dy
    uxyBack = (3.0/2.0*ux[(Nd-1)*N:Nd*N] - 2.0*ux[(Nd-2)*N:(Nd-1)*N]
               + 1.0/2.0*ux[(Nd-3)*N:(Nd-2)*N])/dy
    uxyCent = (-1.0/2.0*ux[0:(Nd-2)*N] + 1.0/2.0*ux[2*N:Nd*N])/dy
    uxy = np.append(np.append(uxyForw, uxyCent), uxyBack)

    dudk = k**4/(12*pi**2)*(3.0/Epi(k, ux)*(1.0/np.tanh(Epi(k, ux)/(2*T)))
                            - 8/Ek(k, g, xx, yy, mu, 1)*(1 - mu/(Eq(k, g, xx)))
                            * (np.tanh(Ek(k, g, xx, yy, mu, 1)/(2*T)))
                            - 8/Ek(k, g, xx, yy, mu, 0)*(1 + mu/(Eq(k, g, xx)))
                            * np.tanh(Ek(k, g, xx, yy, mu, 0)/(2*T)))\
        + k**4*T/(6*pi**2)*func2(k, mu, T, ux, uxx, uy, uyy, uxy, xx, yy)

    # print(k)
    return dudk


if __name__ == "__main__":
    start = time.time()
    lam = 23
    m_lam = 0
    g = 4.8
    k_cutoff = 600
    k_IR = 100
    dk = 0.1
    N_k = (k_cutoff - k_IR)/(dk) + 1
    L = 140.0**2
    N = 50
    Nd = 50
    dx = L / (N)
    dy = L / (Nd)
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, Nd)
    gr = np.linspace(0, L, N)
    xx = x
    yy = np.linspace(0, 0, N)
    for i in range(Nd - 1):
        # create rho vector for functions that only depend on rho
        xx = np.append(xx, x)
    for i in range(Nd - 1):
        # create Delta vector for functions that only depend on Delta
        yy = np.append(yy, np.linspace(y[i + 1], y[i + 1], N))
    for i in range(Nd - 1):
        gr = np.append(gr, np.linspace(y[i + 1], L + y[i + 1], N))
    # k = np.linspace(k_cutoff, k_IR, N_k)
    u0 = 1.0/2.0*m_lam**2*gr + lam/4.0*gr**2

    T_min = 5
    T_max = 200
    N_T = 20
    mu_max = 150
    N_mu = 20
    T_array = np.linspace(T_min, T_max, N_T)
    mu_array = np.linspace(0, mu_max, N_mu)
    expl_sym_br = 1450000*np.sqrt(xx)
    complete_sol = [[None for _ in range(len(mu_array))] for _ in
                    range(len(T_array))]
    min_values = np.empty([len(mu_array), len(T_array)])
    min_values_diq = np.empty([len(mu_array), len(T_array)])

    ode_order = 15
    ode_steps = 500
    ode_integrator = 'vode'
    ode_method = 'bdf'
    ode15s = ode(f)
    ode15s.set_integrator(ode_integrator, method=ode_method, order=ode_order,
                          nsteps=ode_steps)

    # num_cores = multiprocessing.cpu_count()

    for m in range(len(mu_array)):
        for t in range(len(T_array)):
            ode15s.set_initial_value(u0, k_cutoff).set_f_params(N, Nd, g,
                                                                mu_array[m],
                                                                T_array[t], gr,
                                                                dx, xx, yy)
            c = 1
            # input((k, len(k), N_k))
            while ode15s.successful() and c < N_k:
                ode15s.integrate(ode15s.t-dk)
                complete_sol[m][t] = ode15s.y
                c += 1
            print((mu_array[m], T_array[t]))
            c = 0
            sol = complete_sol[m][t]
            argm = np.argmin([sol - expl_sym_br])
            min_values[m, t] = np.sqrt((argm % N)*dx)
            min_values_diq[m, t] = np.sqrt(float(str(float(argm)/N)[0:str(
                float(argm)/N).find(".")])*dy)

    # TODO EXTRACT CORRECT MIN VALUES
    # print("expl", expl_sym_br)
    # print("complete", (complete_sol[0][0] - expl_sym_br)[-1, :])
    print("chiral condensate: "+str(min_values[0, 0]))
    print("diquark condensate: "+str(min_values_diq[-1, 0]))

    end = time.time()
    print(end - start)

    # plt.plot(complete_sol[0][0])
    plt.plot(complete_sol[-1][0])
    plt.show()
    param_list = np.array([lam, m_lam, g, k_cutoff, k_IR, N_k, L, N, dx,
                           T_min, T_max, N_T, mu_max, N_mu, Nd])
    doc = open("optimize.txt", "a")
    doc.write(str((end - start, (ode_integrator, ode_method, ode_order,
                                 ode_steps, lam, k_cutoff, k_IR, dk, N_k, L, N,
                                 Nd, N_T, N_mu, min_values[0, 0])))+"\n")
    dat_name = '2DPhaseDiagramN_T'+str(N_T)+'N_mu'+str(N_mu)
    params_name = dat_name+'params'
    fig_name = dat_name+'.png'
    np.save(dat_name, complete_sol)
    np.save(params_name, param_list)
    mu_ax, T_ax = np.meshgrid(mu_array, T_array)
    fig = plt.figure()
    CS = plt.contourf(mu_ax, T_ax, min_values.T, 16)
    plt.title('Phase Diagram')
    plt.savefig(fig_name)
    plt.show()
    mu_ax, T_ax = np.meshgrid(mu_array, T_array)
    fig = plt.figure()
    CS = plt.contourf(mu_ax, T_ax, min_values_diq.T, 16)
    plt.title('Phase Diagram')
    plt.savefig(fig_name)
    plt.show()
